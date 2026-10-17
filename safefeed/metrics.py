"""Métricas de una ejecución, calculadas a partir de los registros agente + verdad simulada."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _prf(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(_div(2 * p * r, p + r), 4)}


def compute_run_metrics(records: List[dict], review_seconds: float = 45.0) -> dict:
    n = len(records)
    if n == 0:
        return {"n_comments": 0}
    y = np.array([int(r["is_toxic"]) for r in records])
    act = np.array([r["action"] for r in records])
    pred = (act != "publish").astype(int)          # el agente "marca" como dañino si bloquea u oculta
    p_tox = np.array([r["p_toxic"] for r in records])
    lat = np.array([r["latency_ms"] for r in records])

    benign, harmful = (y == 0), (y == 1)
    out: dict = {
        "n_comments": n,
        "toxic_rate": round(float(y.mean()), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "baseline_accuracy_always_publish": round(float(1 - y.mean()), 4),
        **_prf(y, pred),
        "false_block_rate": round(_div(((act == "block") & benign).sum(), benign.sum()), 4),
        "benign_sent_to_review_rate": round(_div(((act == "hide_for_review") & benign).sum(), benign.sum()), 4),
        "harmful_published_rate": round(_div(((act == "publish") & harmful).sum(), harmful.sum()), 4),
        "harmful_blocked_rate": round(_div(((act == "block") & harmful).sum(), harmful.sum()), 4),
        "share_blocked": round(float((act == "block").mean()), 4),
        "share_hidden": round(float((act == "hide_for_review").mean()), 4),
        "share_published": round(float((act == "publish").mean()), 4),
        # Proxy de carga humana: si cada comentario se revisara a mano vs solo los ocultos.
        "human_review_workload_reduction": round(1.0 - float((act == "hide_for_review").mean()), 4),
        "manual_review_seconds_if_all": round(n * review_seconds, 1),
        "agent_review_seconds": round(float((act == "hide_for_review").sum()) * review_seconds, 1),
        "latency_ms_mean": round(float(lat.mean()), 3),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
        "latency_ms_max": round(float(lat.max()), 3),
    }
    if len(set(y.tolist())) == 2:
        out["auc_roc"] = round(float(roc_auc_score(y, p_tox)), 4)
        out["auc_pr"] = round(float(average_precision_score(y, p_tox)), 4)

    # Por fase (antes / después de la deriva de vocabulario).
    by_phase = {}
    for ph in ("pre_drift", "post_drift"):
        m = np.array([r["phase"] == ph for r in records])
        if m.any():
            by_phase[ph] = {"n": int(m.sum()), "accuracy": round(float((pred[m] == y[m]).mean()), 4),
                            **_prf(y[m], pred[m])}
    out["by_phase"] = by_phase

    # Ataques coordinados.
    attack = [r for r in records if r["kind"] == "attack"]
    if attack:
        contained = sum(1 for r in attack if r["action"] != "publish")
        per_campaign = {}
        for cid in sorted({r["campaign_id"] for r in attack}):
            seq = [r for r in attack if r["campaign_id"] == cid]
            leaked_before = 0
            for r in seq:
                if r["action"] == "publish":
                    leaked_before += 1
                else:
                    break
            per_campaign[str(cid)] = {
                "n": len(seq),
                "contained_rate": round(_div(sum(r["action"] != "publish" for r in seq), len(seq)), 4),
                "published_before_first_containment": leaked_before,
                "surge_mode_share": round(_div(sum(bool(r["surge"]) for r in seq), len(seq)), 4),
            }
        out["attacks"] = {
            "n_attack_comments": len(attack),
            "contained_rate": round(_div(contained, len(attack)), 4),
            "leaked": len(attack) - contained,
            "per_campaign": per_campaign,
        }
    return out
