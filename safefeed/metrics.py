"""Métricas separadas del clasificador, el agente y el simulador."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import UtilityConfig
from .utility import (apply_safety_overrides, choose_action, effective_probability,
                      expected_utilities, severity_multiplier)


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _binary_metrics(y: np.ndarray, pred: np.ndarray, scores=None) -> Dict:
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    out = {
        "accuracy": round(_div(tp + tn, len(y)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_div(2 * precision * recall, precision + recall), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "false_positive_rate": round(_div(fp, fp + tn), 4),
        "false_negative_rate": round(_div(fn, fn + tp), 4),
    }
    if scores is not None and len(set(y.tolist())) == 2:
        out["auc_roc"] = round(float(roc_auc_score(y, scores)), 4)
        out["auc_pr"] = round(float(average_precision_score(y, scores)), 4)
    else:
        out["auc_roc"] = None
        out["auc_pr"] = None
    return out


def _phase_metrics(records: List[dict]) -> dict:
    phases = {}
    for phase in ("pre_drift", "post_drift"):
        items = [r for r in records if r.get("phase") == phase]
        if not items:
            continue
        y = np.asarray([int(r["is_toxic"]) for r in items])
        pred = np.asarray([int(r["action"] != "publish") for r in items])
        metrics = _binary_metrics(y, pred)
        phases[phase] = {"n": len(items), **metrics}
    return phases


def _classifier_phase_metrics(records: List[dict], threshold: float) -> dict:
    phases = {}
    for phase in ("pre_drift", "post_drift"):
        items = [r for r in records if r.get("phase") == phase]
        if not items:
            continue
        y = np.asarray([int(r["is_toxic"]) for r in items])
        scores = np.asarray([float(r["p_toxic"]) for r in items])
        phases[phase] = {
            "n": len(items),
            **_binary_metrics(y, (scores >= threshold).astype(int), scores),
        }
    return phases


def compute_run_metrics(records: List[dict], review_seconds: float = 45.0) -> dict:
    if not records:
        return {"n_comments": 0, "classifier": {}, "agent": {}, "simulator": {}}

    y = np.asarray([int(r["is_toxic"]) for r in records])
    actions = np.asarray([r["action"] for r in records])
    scores = np.asarray([float(r["p_toxic"]) for r in records])
    threshold = float(records[0].get("classifier_threshold", 0.5))
    classifier_pred = (scores >= threshold).astype(int)
    agent_pred = (actions != "publish").astype(int)
    latencies = np.asarray([float(r.get("latency_ms", 0.0)) for r in records])
    benign, harmful = (y == 0), (y == 1)
    classifier = _binary_metrics(y, classifier_pred, scores)
    classifier["by_phase"] = _classifier_phase_metrics(records, threshold)
    agent = _binary_metrics(y, agent_pred)
    hidden = actions == "hide_for_review"
    blocked = actions == "block"
    published = actions == "publish"

    agent.update({
        "share_blocked": round(float(blocked.mean()), 4),
        "share_hidden": round(float(hidden.mean()), 4),
        "share_published": round(float(published.mean()), 4),
        "false_block_rate": round(_div(int((blocked & benign).sum()), int(benign.sum())), 4),
        "false_blocks": int((blocked & benign).sum()),
        "benign_sent_to_review_rate": round(_div(int((hidden & benign).sum()), int(benign.sum())), 4),
        "benign_sent_to_review": int((hidden & benign).sum()),
        "harmful_published_rate": round(_div(int((published & harmful).sum()), int(harmful.sum())), 4),
        "harmful_blocked_rate": round(_div(int((blocked & harmful).sum()), int(harmful.sum())), 4),
        "harmful_hidden_rate": round(_div(int((hidden & harmful).sum()), int(harmful.sum())), 4),
        "harmful_published": int((published & harmful).sum()),
        "human_review_workload": int(hidden.sum()),
        "human_review_workload_rate": round(float(hidden.mean()), 4),
        "manual_review_seconds_if_all": round(len(records) * review_seconds, 1),
        "agent_review_seconds": round(int(hidden.sum()) * review_seconds, 1),
        "latency_ms_mean": round(float(latencies.mean()), 3),
        "latency_ms_median": round(float(np.median(latencies)), 3),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 3),
        "latency_ms_max": round(float(latencies.max()), 3),
        "latency_ms_std": round(float(latencies.std()), 3),
    })

    attacks = [r for r in records if r.get("kind") == "attack"]
    attack_metrics = {}
    if attacks:
        contained = [r for r in attacks if r["action"] != "publish"]
        campaigns = {}
        for campaign_id in sorted({r.get("campaign_id") for r in attacks}, key=str):
            seq = [r for r in attacks if r.get("campaign_id") == campaign_id]
            published_before = 0
            for item in seq:
                if item["action"] != "publish":
                    break
                published_before += 1
            campaigns[str(campaign_id)] = {
                "n": len(seq),
                "contained": len(seq) - sum(r["action"] == "publish" for r in seq),
                "contained_rate": round(_div(
                    sum(r["action"] != "publish" for r in seq), len(seq)
                ), 4),
                "published_before_first_containment": published_before,
            }
        rates = np.asarray([c["contained_rate"] for c in campaigns.values()], dtype=float)
        by_phase = {}
        for phase in ("pre_drift", "post_drift"):
            subset = [r for r in attacks if r.get("phase") == phase]
            if subset:
                by_phase[phase] = {
                    "n": len(subset),
                    "contained": sum(r["action"] != "publish" for r in subset),
                    "contained_rate": round(_div(
                        sum(r["action"] != "publish" for r in subset), len(subset)
                    ), 4),
                }
        attack_metrics = {
            "n_attack_comments": len(attacks),
            "contained": len(contained),
            "filtered": len(contained),
            "leaked": len(attacks) - len(contained),
            "contained_rate": round(_div(len(contained), len(attacks)), 4),
            "published_before_first_containment": sum(
                c["published_before_first_containment"] for c in campaigns.values()
            ),
            "by_campaign": {
                "mean": round(float(rates.mean()), 4),
                "std": round(float(rates.std()), 4),
                "min": round(float(rates.min()), 4),
                "max": round(float(rates.max()), 4),
            },
            "per_campaign": campaigns,
            "by_phase": by_phase,
        }

    normal_benign = [
        r for r in records if r.get("kind") != "attack" and not r["is_toxic"]
    ]
    coordination_false_positives = sum(
        float(r.get("coordination", 0.0)) >= 0.5 for r in normal_benign
    )
    api_failures = sum(int(r.get("api_failures_injected", 0)) for r in records)
    safe_failures = sum(
        int(r.get("failure_handled", False)) for r in records
    )
    simulator = {
        "api_errors": sum(int(r.get("api_errors", 0)) for r in records),
        "api_failures_injected": api_failures,
        "safe_failures_handled": safe_failures,
        "safe_failure_handling_rate": (
            round(_div(safe_failures, api_failures), 4) if api_failures else None
        ),
        "coordination_false_positives": coordination_false_positives,
        "coordination_false_positive_rate": round(
            _div(coordination_false_positives, len(normal_benign)), 4
        ),
        "by_phase": _phase_metrics(records),
        "attacks": attack_metrics,
    }
    by_model_version = {}
    for version in sorted({int(r.get("model_version", 0)) for r in records}):
        subset = [r for r in records if int(r.get("model_version", 0)) == version]
        version_y = np.asarray([int(r["is_toxic"]) for r in subset])
        version_scores = np.asarray([float(r["p_toxic"]) for r in subset])
        version_pred = np.asarray([int(r["action"] != "publish") for r in subset])
        by_model_version[str(version)] = {
            "n": len(subset),
            "classifier": _binary_metrics(
                version_y, (version_scores >= threshold).astype(int), version_scores
            ),
            "agent": _binary_metrics(version_y, version_pred),
            "classifier_by_phase": _classifier_phase_metrics(subset, threshold),
            "agent_by_phase": _phase_metrics(subset),
        }
    # Flat aliases retain compatibility; classifier and agent metrics remain explicit above.
    out = {
        "n_comments": len(records),
        "toxic_rate": round(float(y.mean()), 4),
        "baseline_accuracy_always_publish": round(float(1 - y.mean()), 4),
        "classifier_threshold": threshold,
        "classifier": classifier,
        "auc_roc": classifier["auc_roc"],
        "auc_pr": classifier["auc_pr"],
        "agent": agent,
        "simulator": simulator,
        "by_model_version": by_model_version,
        "by_phase": simulator["by_phase"],
        "attacks": attack_metrics,
        "api_errors": simulator["api_errors"],
        "safe_failure_handling_rate": simulator["safe_failure_handling_rate"],
        "coordination_false_positives": coordination_false_positives,
        "coordination_false_positive_rate": simulator["coordination_false_positive_rate"],
    }
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate",
                "false_negative_rate", "false_block_rate", "benign_sent_to_review_rate",
                "harmful_published_rate", "harmful_blocked_rate", "harmful_hidden_rate",
                "human_review_workload", "share_blocked", "share_hidden", "share_published",
                "latency_ms_mean", "latency_ms_median",
                "latency_ms_p95", "latency_ms_max", "latency_ms_std"):
        if key in agent:
            out[key] = agent[key]
    return out


def calibrate_utility_policy(
    validation_records: List[dict],
    penalties: tuple[float, ...] = (-1.5, -2.5, -3.5, -5.0, -7.5, -10.0),
) -> dict:
    """Escoge la política más permisiva que cumple seguridad en validation simulada."""
    if not validation_records:
        raise ValueError("La calibración necesita registros de validación.")
    candidates = []
    for penalty in penalties:
        cfg = replace(UtilityConfig(), publish_harmful=float(penalty))
        policy_records = []
        for record in validation_records:
            p_toxic = float(record["p_toxic"])
            coordination = float(record.get("coordination", 0.0))
            sub_probs = record.get("subcategory_probs", {})
            p_effective = effective_probability(p_toxic, coordination, cfg)
            surge = coordination >= cfg.surge_threshold
            severity = severity_multiplier(sub_probs, cfg)
            action = choose_action(expected_utilities(p_effective, severity, surge, cfg))
            action, _ = apply_safety_overrides(action, p_effective, sub_probs, surge, cfg)
            policy_records.append({
                **record,
                "action": action.value,
                "coordination": coordination,
            })
        metrics = compute_run_metrics(policy_records)
        attack_rate = metrics.get("attacks", {}).get("contained_rate", 1.0)
        agent = metrics["agent"]
        meets = {
            "harmful_published_rate": agent["harmful_published_rate"] <= 0.10,
            "recall": agent["recall"] >= 0.90,
            "false_block_rate": agent["false_block_rate"] <= 0.05,
            "attack_containment_rate": attack_rate >= 0.90,
        }
        candidates.append({
            "publish_harmful": float(penalty),
            "meets_safety_targets": all(meets.values()),
            "targets": meets,
            "metrics": metrics,
        })
    feasible = [candidate for candidate in candidates if candidate["meets_safety_targets"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda candidate: (
                candidate["metrics"]["agent"]["harmful_published_rate"],
                candidate["metrics"]["agent"]["false_block_rate"],
                candidate["metrics"]["agent"]["human_review_workload"],
                -candidate["publish_harmful"],
            ),
        )
        rationale = (
            "Minimiza daño publicado entre políticas factibles; la carga de revisión "
            "solo desempata después de respetar los límites de seguridad."
        )
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate["metrics"]["agent"]["harmful_published_rate"],
                -candidate["metrics"]["agent"]["recall"],
                -candidate["metrics"].get("attacks", {}).get("contained_rate", 1.0),
                candidate["metrics"]["agent"]["false_block_rate"],
            ),
        )
        rationale = "Ninguna política cumple todos los objetivos; se prioriza daño no publicado."
    return {
        "selection_split": "validation_simulation",
        "selected_publish_harmful": selected["publish_harmful"],
        "rationale": rationale,
        "selected_metrics": selected["metrics"],
        "candidates": candidates,
        "test_used_for_selection": False,
    }
