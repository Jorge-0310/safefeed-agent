"""SafeFeedAgent: agente de aprendizaje con componente basado en utilidad.

Ciclo explícito (busca estas cuatro funciones):

    perceive()  -> percepción: limpia el texto y calcula señales de coordinación
    decide()    -> decisión:   probabilidad de toxicidad + utilidad esperada de cada acción
    act()       -> acción:     llama a la API de publicación (bloquear / ocultar / publicar)
    learn       -> receive_feedback()/retrain(): mejora con las decisiones de los moderadores

step() encadena percepción -> decisión -> acción y registra cada fase en el log.
El agente NUNCA recibe la etiqueta verdadera: solo texto + metadatos (ambiente
parcialmente observable) y, después, la retroalimentación de los moderadores.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import LABELS, SUBCATEGORIES, AgentConfig, UtilityConfig
from .coordination import ShortTermMemory
from .logger import NullLogger
from .model import ToxicityModel
from .schemas import (Action, ActionResult, Decision, ModeratorFeedback, Percept,
                      RawComment)
from .text import clean_text, tokenize
from .utility import (choose_action, effective_probability, expected_utilities,
                      apply_safety_overrides, severity_multiplier)


class SafeFeedAgent:
    def __init__(
        self,
        model: ToxicityModel,
        config: AgentConfig = AgentConfig(),
        utility: UtilityConfig = UtilityConfig(),
        logger=None,
    ):
        self.model = model
        self.cfg = config
        self.ucfg = utility
        self.logger = logger or NullLogger()
        # Memoria de corto plazo: comentarios recientes (detección de ataques coordinados).
        self.memory = ShortTermMemory(config.memory_window_s, config.memory_max_items,
                                      config.similarity_threshold)
        # Memoria de largo plazo: el propio modelo (datos etiquetados) + retroalimentación pendiente.
        self._pending: Dict[str, Tuple[str, Action]] = {}
        self._feedback_buffer = deque(maxlen=max(1, config.recent_feedback_max))
        self._feedback_since_retrain = 0
        self.n_retrains = 0
        self.n_feedback = 0

    # ------------------------------------------------------------------ 1) PERCEPCIÓN
    def perceive(self, raw: RawComment) -> Percept:
        clean = clean_text(raw.text)
        tokens = tokenize(raw.text)
        if self.cfg.use_coordination:
            coord = self.memory.features(raw.user_id, raw.timestamp, tokens)
        else:
            coord = {"score": 0.0, "n_similar_recent": 0, "distinct_users_similar": 0,
                     "same_user_last_60s": 0, "window_size": 0}
        required = {"score", "n_similar_recent", "distinct_users_similar",
                    "same_user_last_60s", "window_size"}
        if not isinstance(coord, dict) or not required.issubset(coord):
            raise ValueError("coordination_sensor_missing_data")
        return Percept(raw.comment_id, raw.user_id, raw.timestamp, raw.thread_id,
                       raw.text, clean, tokens, coord)

    # ------------------------------------------------------------------ 2) DECISIÓN
    def decide(self, percept: Percept) -> Decision:
        probs = self.model.predict_proba([percept.text_raw])[0]
        p_toxic = float(probs[0])
        sub = {name: float(p) for name, p in zip(SUBCATEGORIES, probs[1:])}
        severity = severity_multiplier(sub, self.ucfg)
        coord = float(percept.coordination.get("score", 0.0))
        p_eff = effective_probability(p_toxic, coord, self.ucfg)
        surge = coord >= self.ucfg.surge_threshold
        eu = expected_utilities(p_eff, severity, surge, self.ucfg)
        action = choose_action(eu)
        action, safety_rule = apply_safety_overrides(action, p_eff, sub, surge, self.ucfg)

        top = max(sub, key=sub.get)
        subcategory = top if (p_toxic >= 0.3 and sub[top] >= 0.3) else None
        rationale = (
            f"p_toxic={p_toxic:.3f}, coord={coord:.2f}{' (OLA)' if surge else ''}, "
            f"p_ef={p_eff:.3f}, sev={severity:.2f} -> EU "
            + ", ".join(f"{k}={v:+.3f}" for k, v in eu.items())
            + f" => {action.value}"
            + (f" [{safety_rule}]" if safety_rule else "")
        )
        return Decision(
            comment_id=percept.comment_id, action=action, p_toxic=p_toxic,
            subcategory=subcategory, subcategory_probs=sub, severity_mult=severity,
            coordination_score=coord, p_effective=p_eff, surge=surge,
            expected_utilities=eu, rationale=rationale, model_version=self.model.version,
            safety_rule=safety_rule,
        )

    # ------------------------------------------------------------------ 3) ACCIÓN
    def act(self, decision: Decision, percept: Percept, api) -> ActionResult:
        """Herramienta externa: API del sistema de publicación (con reintento)."""
        attempts = 0
        errors = []

        def execute(action: Action):
            nonlocal attempts
            attempts += 1
            if api is None or not callable(getattr(api, "execute", None)):
                return False, "api_unavailable", "api_unavailable"
            try:
                resp = api.execute(action, decision.comment_id, percept.timestamp)
            except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                return False, "api_unavailable", f"api_unavailable:{type(exc).__name__}"
            if (not hasattr(resp, "ok") or not isinstance(resp.ok, bool)
                    or not isinstance(getattr(resp, "status", None), str)):
                return False, "invalid_response", "invalid_api_response"
            if not resp.ok:
                return False, resp.status, f"api_failure:{resp.status}"
            return True, resp.status, None

        actual_action = decision.action
        ok, status, error = False, "api_unavailable", "api_unavailable"
        for _ in range(max(1, 1 + self.cfg.api_retries)):
            ok, status, error = execute(actual_action)
            if error:
                errors.append(error)
            if ok:
                break
        if not ok and actual_action != Action.HIDE:
            actual_action = Action.HIDE
            for _ in range(max(1, 1 + self.cfg.api_retries)):
                ok, fallback_status, fallback_error = execute(Action.HIDE)
                if fallback_error:
                    errors.append(fallback_error)
                status = f"safe_fallback:{fallback_status}"
                if ok:
                    break
        result = ActionResult(
            decision.comment_id, actual_action, ok, status, attempts,
            error=";".join(errors) if errors else None,
        )
        if ok and self.cfg.learning_enabled:
            self._pending[decision.comment_id] = (percept.text_raw, actual_action)
        return result


    # ------------------------------------------------------------------ ciclo completo
    def step(self, raw: RawComment, api) -> Tuple[Percept, Decision, ActionResult]:
        t0 = time.perf_counter()
        errors = self._input_errors(raw)
        if not errors:
            try:
                percept = self.perceive(raw)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc) or "perception_failed")
        if errors:
            percept = self._safe_percept(raw)
            decision = self._safe_decision(percept, ";".join(errors))
        else:
            decision = None
        self.logger.log(
            "percept", percept.timestamp, comment_id=percept.comment_id, user_id=percept.user_id,
            thread_id=percept.thread_id, text_hash=self._text_hash(percept.text_raw),
            coordination=percept.coordination, error=";".join(errors) if errors else None,
        )
        if decision is None:
            decision = self.decide(percept)
        decision.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.logger.log(
            "decision", percept.timestamp, **decision.to_dict(),
            text_hash=self._text_hash(percept.text_raw), coordination=percept.coordination,
        )

        result = self.act(decision, percept, api)
        result.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.logger.log(
            "action", percept.timestamp, comment_id=percept.comment_id,
            text_hash=self._text_hash(percept.text_raw), action=result.action.value,
            ok=result.ok, api_ok=result.ok, status=result.status, attempts=result.attempts,
            latency_ms=result.latency_ms, error=result.error or decision.error,
            model_version=self.model.version, p_toxic=decision.p_toxic,
            subcategory_probs=decision.subcategory_probs, p_effective=decision.p_effective,
            expected_utilities=decision.expected_utilities,
            coordination=percept.coordination,
        )
        # La memoria de corto plazo se actualiza al final (incluye lo bloqueado: fue un intento).
        if not errors:
            self.memory.add(percept.user_id, percept.timestamp, percept.tokens)
        return percept, decision, result

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _input_errors(self, raw: RawComment) -> List[str]:
        if raw is None:
            return ["comment_missing"]
        errors = []
        text = getattr(raw, "text", None)
        if text is None:
            errors.append("text_missing")
        elif not isinstance(text, str):
            errors.append("malformed_comment_text")
        elif not text.strip():
            errors.append("text_empty" if text == "" else "text_whitespace")
        elif len(text) > self.cfg.max_comment_chars:
            errors.append(
                f"comment_too_long:{len(text)}>{self.cfg.max_comment_chars}"
            )
        if not isinstance(getattr(raw, "comment_id", None), str) or not raw.comment_id.strip():
            errors.append("comment_id_missing")
        if not isinstance(getattr(raw, "user_id", None), str) or not raw.user_id.strip():
            errors.append("user_missing")
        if not isinstance(getattr(raw, "thread_id", None), str) or not raw.thread_id.strip():
            errors.append("thread_missing")
        timestamp = getattr(raw, "timestamp", None)
        if not SafeFeedAgent._valid_timestamp(timestamp):
            errors.append("timestamp_invalid")
        return errors

    @staticmethod
    def _valid_timestamp(timestamp) -> bool:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            return False
        try:
            return math.isfinite(float(timestamp))
        except OverflowError:
            return False

    @staticmethod
    def _safe_percept(raw: RawComment) -> Percept:
        def text_value():
            value = getattr(raw, "text", "") if raw is not None else ""
            return value if isinstance(value, str) else ""

        def identifier(name, fallback):
            value = getattr(raw, name, fallback) if raw is not None else fallback
            return value if isinstance(value, str) and value.strip() else fallback

        timestamp = getattr(raw, "timestamp", 0.0) if raw is not None else 0.0
        if not SafeFeedAgent._valid_timestamp(timestamp):
            timestamp = 0.0
        text = text_value()
        return Percept(
            identifier("comment_id", "invalid-comment"),
            identifier("user_id", "unknown-user"),
            float(timestamp),
            identifier("thread_id", "unknown-thread"),
            text, clean_text(text), tokenize(text),
            {"score": 0.0, "n_similar_recent": 0, "distinct_users_similar": 0,
             "same_user_last_60s": 0, "window_size": 0},
        )

    def _safe_decision(self, percept: Percept, error: str) -> Decision:
        return Decision(
            comment_id=percept.comment_id, action=Action.HIDE, p_toxic=0.0,
            subcategory=None, subcategory_probs={name: 0.0 for name in SUBCATEGORIES},
            severity_mult=1.0, coordination_score=0.0, p_effective=0.0,
            surge=False, expected_utilities={}, rationale="Entrada inválida: revisión segura.",
            model_version=self.model.version, safety_rule="invalid_input", error=error,
        )

    # ------------------------------------------------------------------ 4) APRENDIZAJE
    def receive_feedback(self, fb: ModeratorFeedback) -> None:
        self.n_feedback += 1
        pending = self._pending.pop(fb.comment_id, None)
        if pending is None or not self.cfg.learning_enabled:
            self.logger.log("feedback", fb.sim_time, comment_id=fb.comment_id,
                            verdict=fb.verdict, source=fb.source, labels=fb.labels,
                            phase=fb.phase,
                            ground_truth={"labels": fb.labels, "source": "moderator_feedback"})
            return
        text, action = pending if isinstance(pending, tuple) else (pending, Action.HIDE)
        labels = [int(fb.labels.get(k, 0)) for k in LABELS]
        harmful = bool(labels[0])
        predicted_harmful = action != Action.PUBLISH
        classification = (
            "true_positive" if predicted_harmful and harmful else
            "false_positive" if predicted_harmful else
            "false_negative" if harmful else "true_negative"
        )
        self.logger.log(
            "feedback", fb.sim_time, comment_id=fb.comment_id, verdict=fb.verdict,
            source=fb.source, labels=fb.labels, classification=classification,
            phase=fb.phase,
            text_hash=self._text_hash(text), action=action.value,
            ground_truth={"labels": fb.labels, "source": "moderator_feedback"},
            model_version=self.model.version,
        )
        self._feedback_buffer.append((text, labels))
        self._feedback_since_retrain += 1
        if self._feedback_since_retrain >= max(1, self.cfg.retrain_every):
            self.retrain(fb.sim_time)

    def retrain(self, sim_time: Optional[float] = None) -> None:
        """Reentrena el modelo con los ejemplos confirmados/revertidos por moderadores."""
        if not self._feedback_buffer:
            return
        examples = list(self._feedback_buffer)
        texts = [t for t, _ in examples]
        Y = [y for _, y in examples]
        n = len(examples)
        weights = [
            self.cfg.feedback_weight * (0.5 + 0.5 * (i + 1) / n)
            for i in range(n)
        ]
        before_probs = self.model.predict_proba(texts)[:, 0]
        threshold = getattr(self.model, "decision_threshold", 0.5)
        before_accuracy = float(np.mean((before_probs >= threshold) == np.asarray(Y)[:, 0]))
        old_version = self.model.version
        self.model.update(texts, Y, weight=weights)
        after_probs = self.model.predict_proba(texts)[:, 0]
        after_accuracy = float(np.mean((after_probs >= threshold) == np.asarray(Y)[:, 0]))
        self.n_retrains += 1
        self.logger.log(
            "retrain", sim_time, n_new_examples=len(texts), old_version=old_version,
            new_version=self.model.version, fit_seconds=round(self.model.last_fit_seconds, 3),
            total_training_examples=self.model.n_training_examples,
            feedback_buffer_examples=n, recent_examples_higher_weight=True,
            feedback_accuracy_before=round(before_accuracy, 4),
            feedback_accuracy_after=round(after_accuracy, 4),
            model_version=self.model.version,
        )
        self._feedback_since_retrain = 0
        self._feedback_buffer.clear()
