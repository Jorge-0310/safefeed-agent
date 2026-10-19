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

import time
from typing import Dict, List, Optional, Tuple

from .config import LABELS, SUBCATEGORIES, AgentConfig, UtilityConfig
from .coordination import ShortTermMemory
from .logger import NullLogger
from .model import ToxicityModel
from .schemas import (Action, ActionResult, Decision, ModeratorFeedback, Percept,
                      RawComment)
from .text import clean_text, tokenize
from .utility import (choose_action, effective_probability, expected_utilities,
                      severity_multiplier)


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
        self._pending: Dict[str, str] = {}                 # comment_id -> texto, esperando veredicto
        self._feedback_buffer: List[Tuple[str, List[int]]] = []
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

        top = max(sub, key=sub.get)
        subcategory = top if (p_toxic >= 0.3 and sub[top] >= 0.3) else None
        rationale = (
            f"p_toxic={p_toxic:.3f}, coord={coord:.2f}{' (OLA)' if surge else ''}, "
            f"p_ef={p_eff:.3f}, sev={severity:.2f} -> EU "
            + ", ".join(f"{k}={v:+.3f}" for k, v in eu.items())
            + f" => {action.value}"
        )
        return Decision(
            comment_id=percept.comment_id, action=action, p_toxic=p_toxic,
            subcategory=subcategory, subcategory_probs=sub, severity_mult=severity,
            coordination_score=coord, p_effective=p_eff, surge=surge,
            expected_utilities=eu, rationale=rationale, model_version=self.model.version,
        )

    # ------------------------------------------------------------------ 3) ACCIÓN
    def act(self, decision: Decision, percept: Percept, api) -> ActionResult:
        """Herramienta externa: API del sistema de publicación con reintentos."""

        attempts = 0
        resp = None

        for attempt in range(1, 1 + self.cfg.api_retries + 1):
            attempts += 1

            resp = api.execute(
                decision.action,
                decision.comment_id,
                percept.timestamp,
            )

            if resp.ok:
                break

            if attempt <= self.cfg.api_retries:
                self.logger.log(
                    "action_retry",
                    percept.timestamp,
                    comment_id=decision.comment_id,
                    action=decision.action.value,
                    attempt=attempt,
                    status=resp.status,
                )

        if not resp.ok:
            self.logger.log(
                "action_failed",
                percept.timestamp,
                comment_id=decision.comment_id,
                action=decision.action.value,
                attempts=attempts,
                status=resp.status,
            )

        result = ActionResult(
            decision.comment_id,
            decision.action,
            resp.ok,
            resp.status,
            attempts,
        )

        # Solo se aprende de acciones que sí fueron ejecutadas.
        if (
            result.ok
            and decision.action != Action.PUBLISH
            and self.cfg.learning_enabled
        ):
            self._pending[decision.comment_id] = percept.text_raw

        return result


    # ------------------------------------------------------------------ ciclo completo
    def step(self, raw: RawComment, api) -> Tuple[Percept, Decision, ActionResult]:
        t0 = time.perf_counter()
        percept = self.perceive(raw)
        self.logger.log(
            "percept", raw.timestamp, comment_id=raw.comment_id, user_id=raw.user_id,
            thread_id=raw.thread_id, text=raw.text, coordination=percept.coordination,
        )
        decision = self.decide(percept)
        decision.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.logger.log("decision", raw.timestamp, **decision.to_dict())

        result = self.act(decision, percept, api)
        self.logger.log(
            "action", raw.timestamp, comment_id=raw.comment_id, action=result.action.value,
            ok=result.ok, status=result.status, attempts=result.attempts,
            total_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )
        # La memoria de corto plazo se actualiza al final (incluye lo bloqueado: fue un intento).
        self.memory.add(raw.user_id, raw.timestamp, percept.tokens)
        return percept, decision, result

    # ------------------------------------------------------------------ 4) APRENDIZAJE
    def receive_feedback(self, fb: ModeratorFeedback) -> None:
        self.n_feedback += 1
        self.logger.log("feedback", fb.sim_time, comment_id=fb.comment_id, verdict=fb.verdict,
                        source=fb.source, labels=fb.labels)
        text = self._pending.pop(fb.comment_id, None)
        if text is None or not self.cfg.learning_enabled:
            return
        self._feedback_buffer.append((text, [int(fb.labels.get(k, 0)) for k in LABELS]))
        if len(self._feedback_buffer) >= self.cfg.retrain_every:
            self.retrain(fb.sim_time)

    def retrain(self, sim_time: Optional[float] = None) -> None:
        """Reentrena el modelo con los ejemplos confirmados/revertidos por moderadores."""
        if not self._feedback_buffer:
            return
        texts = [t for t, _ in self._feedback_buffer]
        Y = [y for _, y in self._feedback_buffer]
        old_version = self.model.version
        self.model.update(texts, Y, weight=self.cfg.feedback_weight)
        self.n_retrains += 1
        self.logger.log(
            "retrain", sim_time, n_new_examples=len(texts), old_version=old_version,
            new_version=self.model.version, fit_seconds=round(self.model.last_fit_seconds, 3),
            total_training_examples=self.model.n_training_examples,
        )
        self._feedback_buffer = []
