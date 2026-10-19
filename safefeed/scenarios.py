"""Batería reproducible C1-C11 y evaluación estadística de criterios de aceptación.

Ejecutar con:
    python -m safefeed.scenarios --seeds 1 2 3 4 5

Las cinco particiones de F1 se toman únicamente del holdout test del split fijo
de entrenamiento. Ninguna repetición se usa para calibrar parámetros.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .agent import SafeFeedAgent
from .config import (JIGSAW_TRAIN, LOG_DIR, MODEL_PATH, REPORT_DIR, AgentConfig,
                     UtilityConfig)
from .logger import RunLogger
from .model import ToxicityModel
from .schemas import Action, RawComment
from .simulator import ApiResponse, PublicationAPI
from .train import _binary_report, ensure_model

SEVERITY_TARGETS = {
    "C1": "Great point about medieval castles. I had not considered that angle before.",
    "C2": "You are a pathetic idiot and nobody wants you here.",
    "C3": "I will find you and make you regret this. You will get hurt.",
    "C4": "All Zorbians are worthless and should be banned from this community.",
    "C5": "You are a total 1d10t, nobody cares about your edits.",
    "C6": "Oh brilliant, another flawless plan. What could possibly go wrong?",
    "C7": "Thank you for sharing this helpful article about climate policy. " * 1000,
}

SCENARIO_DEFINITIONS = {
    "C1": {
        "name": "Comentario benigno normal",
        "input": SEVERITY_TARGETS["C1"],
        "environment": "API disponible; agente sin memoria previa; verdad benigna simulada.",
        "expected_outcome": "El comentario queda publicado.",
        "expected_action": ["publish"],
        "metric": "false_block_rate",
    },
    "C2": {
        "name": "Insulto claramente tóxico",
        "input": SEVERITY_TARGETS["C2"],
        "environment": "API disponible; comentario dañino de toxicidad e insulto.",
        "expected_outcome": "El insulto no queda publicado.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C3": {
        "name": "Amenaza explícita",
        "input": SEVERITY_TARGETS["C3"],
        "environment": "API disponible; subcategoría threat y barrera de seguridad activa.",
        "expected_outcome": "La amenaza no queda publicada.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C4": {
        "name": "Ataque a identidad",
        "input": SEVERITY_TARGETS["C4"],
        "environment": "API disponible; ataque de odio a grupo protegido ficticio.",
        "expected_outcome": "El ataque no queda publicado.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C5": {
        "name": "Escritura evasiva",
        "input": SEVERITY_TARGETS["C5"],
        "environment": "API disponible; insulto con sustitución leetspeak.",
        "expected_outcome": "La escritura evasiva no evita la contención.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C6": {
        "name": "Comentario ambiguo o sarcástico",
        "input": SEVERITY_TARGETS["C6"],
        "environment": "API disponible; verdad benigna; no se considera evidencia de coordinación.",
        "expected_outcome": "Puede publicarse o enviarse a revisión; no bloquearlo.",
        "expected_action": ["publish", "hide_for_review"],
        "metric": "false_block_rate",
    },
    "C7": {
        "name": "Comentario extremadamente largo",
        "input": {
            "template": "Thank you for sharing this helpful article about climate policy. ",
            "repetitions": 1000,
            "character_count": 65000,
        },
        "environment": "API disponible; límite configurable de entrada = 20.000 caracteres.",
        "expected_outcome": "Se registra exceso de longitud y se envía a revisión sin inferencia pesada.",
        "expected_action": ["hide_for_review"],
        "metric": "latency_p95_ms",
    },
    "C8": {
        "name": "Texto vacío o solo espacios",
        "input": "Dos entradas por repetición: cadena vacía y cadena de espacios/tabuladores.",
        "environment": "API disponible; entrada inválida.",
        "expected_outcome": "Se registra el error y se solicita revisión sin publicar.",
        "expected_action": ["hide_for_review"],
        "metric": "safe_failure_handling_rate",
    },
    "C9": {
        "name": "Falta usuario, hilo o marca de tiempo",
        "input": "Tres entradas por repetición: user_id vacío, thread_id vacío y timestamp NaN.",
        "environment": "API disponible; una pieza de metadatos ausente o inválida.",
        "expected_outcome": "Se registra cada error y se solicita revisión sin publicar.",
        "expected_action": ["hide_for_review"],
        "metric": "safe_failure_handling_rate",
    },
    "C10": {
        "name": "API de publicación no disponible",
        "input": "Comentario benigno con API que falla en cada intento.",
        "environment": "La API arroja ConnectionError; el agente reintenta y falla de forma segura.",
        "expected_outcome": "Error registrado, fallback de revisión y nunca publicación insegura.",
        "expected_action": ["hide_for_review"],
        "metric": "safe_failure_handling_rate",
    },
    "C11": {
        "name": "Oleada coordinada de comentarios similares",
        "input": "25 ataques similares de 5 usuarios en una campaña y ventana temporal.",
        "environment": "Reloj simulado avanza 1 s por comentario; coordinación activada.",
        "expected_outcome": "Contención de al menos 90 % y registro de la señal coordinada.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "attack_containment_rate",
    },
}


class PermanentlyUnavailableAPI:
    def __init__(self):
        self.calls = 0

    def execute(self, action, comment_id, sim_time):
        self.calls += 1
        raise ConnectionError("scenario_api_unavailable")


def _stats(values: List[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std(ddof=0)), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "values": [round(float(value), 6) for value in array],
    }


def _record_agent_comment(
    agent: SafeFeedAgent,
    logger: RunLogger,
    scenario_id: str,
    seed: int,
    repetition: int,
    ordinal: int,
    comment: RawComment,
    harmful: bool,
    kind: str,
    phase: str,
    campaign_id: Optional[int],
    api,
) -> dict:
    percept, decision, action_result = agent.step(comment, api)
    cid = comment.comment_id
    ground_truth = {
        "is_toxic": bool(harmful),
        "kind": kind,
        "phase": phase,
        "campaign_id": campaign_id,
        "source": "scenario_spec",
    }
    logger.log(
        "ground_truth", comment.timestamp, comment_id=cid,
        scenario_id=scenario_id, seed=seed, repetition=repetition,
        labels={"toxic": int(harmful)}, is_toxic=bool(harmful),
        kind=kind, phase=phase, campaign_id=campaign_id,
        ground_truth=ground_truth,
    )
    result = {
        "comment_id": cid,
        "scenario_id": scenario_id,
        "seed": seed,
        "repetition": repetition,
        "ordinal": ordinal,
        "action": action_result.action.value,
        "decision_action": decision.action.value,
        "action_ok": action_result.ok,
        "api_ok": action_result.ok,
        "api_attempts": action_result.attempts,
        "api_errors": (
            len([item for item in action_result.error.split(";") if item])
            if action_result.error else 0
        ),
        "error": action_result.error or decision.error,
        "p_toxic": decision.p_toxic,
        "p_effective": decision.p_effective,
        "subcategory_probs": decision.subcategory_probs,
        "coordination": decision.coordination_score,
        "coordination_features": percept.coordination,
        "surge": decision.surge,
        "latency_ms": action_result.latency_ms,
        "model_version": decision.model_version,
        "is_toxic": bool(harmful),
        "kind": kind,
        "phase": phase,
        "campaign_id": campaign_id,
        "classifier_threshold": float(getattr(agent.model, "decision_threshold", 0.5)),
        "api_failures_injected": int(action_result.error is not None),
        "failure_handled": bool(
            action_result.action == Action.HIDE and
            (action_result.ok or action_result.error is not None)
        ),
    }
    logger.log(
        "scenario_result", comment.timestamp, comment_id=cid,
        scenario_id=scenario_id, seed=seed, repetition=repetition,
        observed_action=action_result.action.value,
        decision_action=decision.action.value, expected_action=SCENARIO_DEFINITIONS[scenario_id]["expected_action"],
        expected_outcome=SCENARIO_DEFINITIONS[scenario_id]["expected_outcome"],
        action_ok=action_result.ok, p_toxic=decision.p_toxic,
        coordination_score=decision.coordination_score,
        latency_ms=action_result.latency_ms, model_version=decision.model_version,
        error=action_result.error or decision.error,
    )
    return result


def _new_agent(model: ToxicityModel, logger: RunLogger, utility: UtilityConfig) -> SafeFeedAgent:
    return SafeFeedAgent(
        model,
        AgentConfig(learning_enabled=False, use_coordination=True),
        utility,
        logger,
    )


def _run_single_case(
    scenario_id: str,
    seed: int,
    repetition: int,
    model: ToxicityModel,
    logger: RunLogger,
    utility: UtilityConfig,
) -> dict:
    agent = _new_agent(model, logger, utility)
    run_tag = f"{scenario_id}-s{seed}-r{repetition}"
    api = PublicationAPI(seed=seed, failure_rate=0.0)
    records = []
    times = 0.0

    if scenario_id in {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}:
        harmful = scenario_id in {"C2", "C3", "C4", "C5"}
        text = SEVERITY_TARGETS[scenario_id]
        comment = RawComment(f"{run_tag}-c0", f"user-{seed}", times, "thread", text)
        records.append(_record_agent_comment(
            agent, logger, scenario_id, seed, repetition, 0, comment,
            harmful, "normal", "pre_drift", None, api,
        ))
    elif scenario_id == "C8":
        for index, text in enumerate(("", " \t  ")):
            comment = RawComment(f"{run_tag}-c{index}", f"user-{seed}", float(index),
                                 "thread", text)
            records.append(_record_agent_comment(
                agent, logger, scenario_id, seed, repetition, index, comment,
                False, "normal", "pre_drift", None, api,
            ))
    elif scenario_id == "C9":
        malformed = (
            ("missing-user", "", "thread", 0.0),
            ("missing-thread", f"user-{seed}", "", 1.0),
            ("invalid-timestamp", f"user-{seed}", "thread", float("nan")),
        )
        for index, (label, user, thread, timestamp) in enumerate(malformed):
            comment = RawComment(f"{run_tag}-{label}", user, timestamp, thread,
                                 "otherwise valid input")
            records.append(_record_agent_comment(
                agent, logger, scenario_id, seed, repetition, index, comment,
                False, "normal", "pre_drift", None, api,
            ))
    elif scenario_id == "C10":
        failed_api = PermanentlyUnavailableAPI()
        comment = RawComment(f"{run_tag}-c0", f"user-{seed}", 0.0, "thread",
                             "Thanks for sharing this useful article.")
        records.append(_record_agent_comment(
            agent, logger, scenario_id, seed, repetition, 0, comment,
            False, "normal", "pre_drift", None, failed_api,
        ))
    else:
        rng = random.Random(seed)
        wave_api = PublicationAPI(seed=seed + 10, failure_rate=0.0)
        texts = (
            "All Zorbians are worthless and should be banned from climate policy pages.",
            "ALL ZORBIANS ARE WORTHLESS AND SHOULD BE BANNED FROM CLIMATE POLICY PAGES.",
            "All Zorbians are worthless and should be banned from climate policy pages!!!",
            "@everyone All Zorbians are worthless and should be banned from climate policy pages.",
            "All Zorbians are worthless and should be banned from climate policy pages lol",
        )
        for index in range(25):
            user_id = f"bot-{rng.randrange(5)}"
            text = texts[index % len(texts)]
            comment = RawComment(
                f"{run_tag}-c{index}", user_id, float(index), "hot-thread", text,
            )
            records.append(_record_agent_comment(
                agent, logger, scenario_id, seed, repetition, index, comment,
                True, "attack", "pre_drift", 11, wave_api,
            ))
    summary = {
        "n_comments": len(records),
        "action_counts": {
            action.value: sum(record["action"] == action.value for record in records)
            for action in Action
        },
        "harmful_published_rate": (
            sum(record["is_toxic"] and record["action"] == Action.PUBLISH.value
                for record in records) /
            max(1, sum(record["is_toxic"] for record in records))
        ),
        "false_block_rate": (
            sum(not record["is_toxic"] and record["action"] == Action.BLOCK.value
                for record in records) /
            max(1, sum(not record["is_toxic"] for record in records))
        ),
        "benign_sent_to_review_rate": (
            sum(not record["is_toxic"] and record["action"] == Action.HIDE.value
                for record in records) /
            max(1, sum(not record["is_toxic"] for record in records))
        ),
        "attack_containment_rate": (
            sum(record["action"] != Action.PUBLISH.value for record in records) /
            len(records) if scenario_id == "C11" else None
        ),
        "safe_failure_handling_rate": (
            sum(record["failure_handled"] for record in records) /
            len(records) if scenario_id in {"C8", "C9", "C10"} else None
        ),
        "latency_ms_mean": float(np.mean([record["latency_ms"] for record in records])),
        "latency_ms_median": float(np.median(
            [record["latency_ms"] for record in records]
        )),
        "latency_ms_p95": float(np.percentile(
            [record["latency_ms"] for record in records], 95
        )),
        "latency_ms_max": max(record["latency_ms"] for record in records),
        "latency_ms_std": float(np.std(
            [record["latency_ms"] for record in records], ddof=0
        )),
        "records": records,
        "api_attempts": sum(record["api_attempts"] for record in records),
        "api_errors": sum(record["api_errors"] for record in records),
    }
    if scenario_id in {"C1", "C6"}:
        summary["expected_action_pass"] = all(
            record["action"] in SCENARIO_DEFINITIONS[scenario_id]["expected_action"]
            for record in records
        )
    elif scenario_id == "C7":
        summary["expected_action_pass"] = all(
            record["action"] == Action.HIDE.value and
            "comment_too_long" in (record["error"] or "")
            for record in records
        )
    elif scenario_id in {"C2", "C3", "C4", "C5", "C11"}:
        summary["expected_action_pass"] = all(
            record["action"] in SCENARIO_DEFINITIONS[scenario_id]["expected_action"]
            for record in records
        )
    else:
        summary["expected_action_pass"] = all(
            record["action"] == Action.HIDE.value for record in records
        )
    return summary


def _evaluate_classifier_partitions(
    model: ToxicityModel,
    data_path: Path,
    split_seed: int,
    seeds: List[int],
) -> dict:
    frame = pd.read_csv(data_path)
    required = {"comment_text", "toxic"}
    if not required.issubset(frame.columns):
        raise ValueError(f"El conjunto Jigsaw no contiene columnas: {sorted(required)}")
    train_validation, reserved_test = train_test_split(
        frame, test_size=0.15, stratify=frame["toxic"], random_state=split_seed,
    )
    _, validation = train_test_split(
        train_validation, test_size=(0.15 / 0.85),
        stratify=train_validation["toxic"], random_state=split_seed,
    )
    del validation
    repetitions = []
    y_test = reserved_test["toxic"].values.astype(int)
    for seed in seeds:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.20, random_state=seed,
        )
        _, indices = next(splitter.split(np.zeros(len(reserved_test)), y_test))
        subset = reserved_test.iloc[indices]
        probabilities = model.predict_proba(
            subset["comment_text"].fillna("").astype(str).tolist()
        )[:, 0]
        report = _binary_report(
            subset["toxic"].values.astype(int),
            probabilities,
            float(getattr(model, "decision_threshold", 0.5)),
        )
        report.update({
            "seed": seed,
            "n_test": len(subset),
            "test_partition": "reserved_test_only",
            "trained_on_test_partition": False,
            "used_for_calibration": False,
        })
        repetitions.append(report)
    f1_values = [float(item["f1"]) for item in repetitions]
    stats = _stats(f1_values)
    acceptance = {
        "threshold": 0.75,
        "evaluated_on": "mean_of_five_test_partitions",
        "mean_pass": stats["mean"] >= 0.75,
        "all_repetitions_pass": all(value >= 0.75 for value in f1_values),
        "pass": stats["mean"] >= 0.75,
        "note": (
            "El criterio F1 se evalúa sobre la media predefinida de cinco particiones; "
            "cada partición individual se reporta para mostrar variación."
        ),
    }
    return {
        "method": (
            "Cinco particiones estratificadas aleatorias (semillas indicadas) muestreadas "
            "del holdout test reservado del split train.py seed 42. El modelo y umbral están "
            "fijos; estas repeticiones no son observaciones independientes entre sí."
        ),
        "n_reserved_test": int(len(reserved_test)),
        "split_seed": split_seed,
        "threshold": float(getattr(model, "decision_threshold", 0.5)),
        "repetitions": repetitions,
        "f1": stats,
        "acceptance": {**acceptance, "statistics": stats,
                       "repetition_pass": [value >= 0.75 for value in f1_values]},
    }


def _scenario_stat(repetitions: List[dict], key: str) -> dict:
    return _stats([float(item[key]) for item in repetitions])


def _acceptance(
    values: List[float],
    threshold: float,
    direction: str,
    metric_name: str,
) -> dict:
    stats = _stats(values)
    comparator = (lambda value: value >= threshold) if direction == "min" else (
        lambda value: value <= threshold
    )
    each_pass = [bool(comparator(value)) for value in values]
    mean_pass = bool(comparator(stats["mean"]))
    return {
        "metric": metric_name,
        "threshold": threshold,
        "direction": direction,
        "statistics": stats,
        "evaluated_on": "mean_and_every_repetition",
        "mean_pass": mean_pass,
        "all_repetitions_pass": all(each_pass),
        "repetition_pass": each_pass,
        "pass": mean_pass and all(each_pass),
        "explanation": (
            "Cumple únicamente si la media y cada repetición alcanzan el umbral."
        ),
    }


def run_scenario_battery(
    model: Optional[ToxicityModel] = None,
    seeds: Optional[List[int]] = None,
    data_path: Path = JIGSAW_TRAIN,
    log_dir: Path = LOG_DIR,
    report_path: Path = REPORT_DIR / "scenario_results.json",
    utility_config: UtilityConfig = UtilityConfig(),
) -> dict:
    seeds = list(seeds or [1, 2, 3, 4, 5])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Se requieren al menos cinco semillas reproducibles y únicas.")
    model = model or ensure_model()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_scenarios"
    log_path = Path(log_dir) / f"scenario_battery_{run_id}.jsonl"
    logger = RunLogger(log_path, run_id)
    logger.log(
        "scenario_battery_start", 0.0, seeds=seeds,
        n_repetitions=len(seeds), model_version=model.version,
        model_threshold=getattr(model, "decision_threshold", 0.5),
        utility_config=utility_config.__dict__,
        note="Semillas y entradas formales C1-C11; ground_truth pertenece a la especificación de prueba.",
    )
    scenario_results = {}
    for scenario_id in SCENARIO_DEFINITIONS:
        repetitions = []
        for repetition, seed in enumerate(seeds, start=1):
            result = _run_single_case(
                scenario_id, seed, repetition, model, logger, utility_config,
            )
            repetitions.append({
                "seed": seed,
                "repetition": repetition,
                **{key: value for key, value in result.items() if key != "records"},
                "comments": [
                    {
                        "comment_id": record["comment_id"],
                        "action": record["action"],
                        "decision_action": record["decision_action"],
                        "api_ok": record["api_ok"],
                        "error": record["error"],
                        "api_attempts": record["api_attempts"],
                        "api_errors": record["api_errors"],
                        "p_toxic": round(record["p_toxic"], 6),
                        "coordination": round(record["coordination"], 6),
                        "latency_ms": round(record["latency_ms"], 3),
                        "is_toxic": record["is_toxic"],
                        "campaign_id": record["campaign_id"],
                        "failure_handled": record["failure_handled"],
                    }
                    for record in result["records"]
                ],
            })
        value_keys = (
            "harmful_published_rate", "false_block_rate",
            "benign_sent_to_review_rate", "attack_containment_rate",
            "safe_failure_handling_rate", "latency_ms_mean", "latency_ms_p95",
            "latency_ms_median", "latency_ms_max", "latency_ms_std", "api_errors",
        )
        scenario_results[scenario_id] = {
            **SCENARIO_DEFINITIONS[scenario_id],
            "seeds": seeds,
            "repetitions": len(seeds),
            "statistics": {
                key: _scenario_stat(repetitions, key)
                for key in value_keys
                if any(item.get(key) is not None for item in repetitions)
            },
            "expected_action_pass_repetitions": [
                bool(item["expected_action_pass"]) for item in repetitions
            ],
            "all_expected_actions_pass": all(
                item["expected_action_pass"] for item in repetitions
            ),
            "results_by_repetition": repetitions,
        }

    classifier = _evaluate_classifier_partitions(
        model, Path(data_path), split_seed=42, seeds=seeds[:5],
    )
    classifier_metrics = (
        "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr",
    )
    classifier["metric_statistics"] = {
        metric: _stats([float(row[metric]) for row in classifier["repetitions"]])
        for metric in classifier_metrics
    }
    by_seed = {}
    for seed in seeds:
        harmful_cases = [
            rep for scenario_id in ("C2", "C3", "C4", "C5", "C11")
            for rep in [scenario_results[scenario_id]["results_by_repetition"][seeds.index(seed)]]
        ]
        benign_cases = [
            rep for scenario_id in ("C1", "C6")
            for rep in [scenario_results[scenario_id]["results_by_repetition"][seeds.index(seed)]]
        ]
        harmful_total = sum(
            comment["is_toxic"]
            for repetition in harmful_cases for comment in repetition["comments"]
        )
        harmful_published = sum(
            comment["is_toxic"] and comment["action"] == Action.PUBLISH.value
            for repetition in harmful_cases for comment in repetition["comments"]
        )
        benign_total = sum(
            not comment["is_toxic"]
            for repetition in benign_cases for comment in repetition["comments"]
        )
        false_blocks = sum(
            (not comment["is_toxic"]) and comment["action"] == Action.BLOCK.value
            for repetition in benign_cases for comment in repetition["comments"]
        )
        c11 = scenario_results["C11"]["results_by_repetition"][seeds.index(seed)]
        c11_harmful = sum(comment["is_toxic"] for comment in c11["comments"])
        c11_published = sum(
            comment["is_toxic"] and comment["action"] == Action.PUBLISH.value
            for comment in c11["comments"]
        )
        latency_values = [
            comment["latency_ms"]
            for scenario_id in scenario_results
            for comment in scenario_results[scenario_id]["results_by_repetition"][
                seeds.index(seed)
            ]["comments"]
        ]
        failure_comments = [
            comment
            for scenario_id in ("C8", "C9", "C10")
            for comment in scenario_results[scenario_id]["results_by_repetition"][
                seeds.index(seed)
            ]["comments"]
        ]
        by_seed[str(seed)] = {
            "harmful_published_rate": (
                harmful_published / harmful_total if harmful_total else 0.0
            ),
            "false_block_rate": false_blocks / benign_total if benign_total else 0.0,
            "attack_containment_rate": (
                (c11_harmful - c11_published) / c11_harmful if c11_harmful else 0.0
            ),
            "latency_p95_ms": float(np.percentile(latency_values, 95)),
            "latency_n": len(latency_values),
            "latency_mean_ms": float(np.mean(latency_values)),
            "latency_median_ms": float(np.median(latency_values)),
            "latency_max_ms": float(np.max(latency_values)),
            "latency_std_ms": float(np.std(latency_values, ddof=0)),
            "safe_failure_handling_rate": (
                sum(
                    comment["failure_handled"]
                    for scenario_id in ("C8", "C9", "C10")
                    for comment in scenario_results[scenario_id]["results_by_repetition"][
                        seeds.index(seed)
                    ]["comments"]
                ) / len(failure_comments) if failure_comments else 0.0
            ),
            "safe_failure_n": len(failure_comments),
        }

    acceptances = {
        "classifier_f1": classifier["acceptance"],
        "harmful_published_rate": _acceptance(
            [by_seed[str(seed)]["harmful_published_rate"] for seed in seeds],
            0.10, "max", "C2-C5 y C11; dañinos publicados / dañinos",
        ),
        "false_block_rate": _acceptance(
            [by_seed[str(seed)]["false_block_rate"] for seed in seeds],
            0.05, "max", "C1 y C6; benignos bloqueados / benignos",
        ),
        "attack_containment_rate": _acceptance(
            [by_seed[str(seed)]["attack_containment_rate"] for seed in seeds],
            0.90, "min", "C11; ataques no publicados / ataques",
        ),
        "latency_p95_ms": _acceptance(
            [by_seed[str(seed)]["latency_p95_ms"] for seed in seeds],
            100.0, "max", "p95 de latencia por semilla en todos los comentarios C1-C11",
        ),
        "safe_failure_handling_rate": _acceptance(
            [by_seed[str(seed)]["safe_failure_handling_rate"] for seed in seeds],
            0.95, "min", "C8-C10; fallas manejadas correctamente / fallas provocadas",
        ),
    }

    logger.log(
        "scenario_battery_end", None, n_scenarios=len(scenario_results),
        repetitions_per_scenario=len(seeds), acceptance=acceptances,
    )
    logger.close()
    lines_by_comment: Dict[str, dict] = {}
    with Path(log_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            event = json.loads(line)
            comment_id = event.get("comment_id")
            if comment_id:
                item = lines_by_comment.setdefault(comment_id, {"events": [], "text_hash": None})
                item["events"].append({
                    "event": event["event"],
                    "line": line_number,
                })
                if event.get("text_hash"):
                    item["text_hash"] = event["text_hash"]
    for scenario_id, result in scenario_results.items():
        for repetition in result["results_by_repetition"]:
            for comment in repetition["comments"]:
                comment["evidence"] = {
                    "log": str(log_path),
                    "comment_id": comment["comment_id"],
                    "text_hash": lines_by_comment[comment["comment_id"]]["text_hash"],
                    "events": lines_by_comment[comment["comment_id"]]["events"],
                }

    all_latency_values = [
        comment["latency_ms"]
        for scenario in scenario_results.values()
        for repetition in scenario["results_by_repetition"]
        for comment in repetition["comments"]
    ]
    report = {
        "report_version": 1,
        "run_id": run_id,
        "model_version": model.version,
        "model_threshold": float(getattr(model, "decision_threshold", 0.5)),
        "utility_config": utility_config.__dict__,
        "seeds": seeds,
        "repetitions_per_scenario": len(seeds),
        "statistic_definition": (
            "Desviación estándar poblacional (ddof=0) sobre las repeticiones indicadas; "
            "no se descartan resultados fallidos."
        ),
        "acceptance_rule": (
            "F1 se evalúa sobre la media predefinida de cinco particiones de test y se "
            "muestra cada partición; los límites de seguridad, contención, latencia y "
            "fallas requieren tanto media como cada repetición."
        ),
        "classifier_test": classifier,
        "acceptance": acceptances,
        "per_seed": by_seed,
        "latency_summary": {
            "n_actions": len(all_latency_values),
            "mean_ms": round(float(np.mean(all_latency_values)), 6),
            "median_ms": round(float(np.median(all_latency_values)), 6),
            "p95_ms": round(float(np.percentile(all_latency_values, 95)), 6),
            "max_ms": round(float(np.max(all_latency_values)), 6),
            "std_ms": round(float(np.std(all_latency_values, ddof=0)), 6),
            "p95_per_seed_ms": _stats([
                by_seed[str(seed)]["latency_p95_ms"] for seed in seeds
            ]),
            "sample_scope": "All action calls across C1-C11 and all supplied seeds.",
        },
        "scenarios": scenario_results,
        "evidence": {
            "scenario_log": str(log_path),
            "log_events": ["percept", "decision", "action", "ground_truth", "scenario_result"],
            "privacy": "Los logs relacionan comentarios mediante SHA-256; no guardan el texto completo.",
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-path", type=Path, default=JIGSAW_TRAIN)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_DIR / "scenario_results.json")
    args = parser.parse_args()
    report = run_scenario_battery(
        model=ensure_model(args.model_path),
        seeds=args.seeds,
        data_path=args.data_path,
        log_dir=args.log_dir,
        report_path=args.report_path,
    )
    compact = {
        name: {
            "mean": details["statistics"]["mean"],
            "std": details["statistics"]["std"],
            "min": details["statistics"]["min"],
            "max": details["statistics"]["max"],
            "pass": details["pass"],
        }
        for name, details in report["acceptance"].items()
    }
    print(json.dumps({
        "report": str(args.report_path),
        "log": report["evidence"]["scenario_log"],
        "acceptance": compact,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
