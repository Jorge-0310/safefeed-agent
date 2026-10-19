"""Entrena y calibra sin fuga de información; evalúa una sola vez en test.

    python -m safefeed.train
    python -m safefeed.train --max-rows 40000

Usa división estratificada 70/15/15 (seed 42 por defecto). La validación decide
entre LogisticRegression sin peso y class_weight="balanced", y escoge el umbral.
El test no participa en esas elecciones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .config import (CLASSIFIER_THRESHOLDS, JIGSAW_TRAIN, LABELS, MODEL_PATH,
                     REPORT_DIR, UtilityConfig)
from .data import load_training_data
from .model import ToxicityModel
from .utility import utility_action_regions


def _binary_report(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    result = {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        "false_positive_rate": round(
            float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else 0.0, 4
        ),
        "false_negative_rate": round(
            float(cm[1, 0] / cm[1].sum()) if cm[1].sum() else 0.0, 4
        ),
    }
    if len(np.unique(y)) == 2:
        result["auc_roc"] = round(float(roc_auc_score(y, probabilities)), 4)
        result["auc_pr"] = round(float(average_precision_score(y, probabilities)), 4)
    else:
        result["auc_roc"] = None
        result["auc_pr"] = None
    return result


def calibrate_threshold(y_validation: np.ndarray, probabilities: np.ndarray) -> dict:
    """Prioriza FNR <=10%; dentro de ese objetivo limita FPR a 5% y maximiza recall."""
    candidates = [
        _binary_report(y_validation, probabilities, threshold)
        for threshold in CLASSIFIER_THRESHOLDS
    ]
    acceptable_fp = [m for m in candidates if m["false_positive_rate"] <= 0.05]
    safe = [
        m for m in acceptable_fp
        if m["false_negative_rate"] <= 0.10
    ]
    if safe:
        selected = max(safe, key=lambda m: (m["recall"], m["f1"], m["threshold"]))
    elif acceptable_fp:
        selected = max(
            acceptable_fp,
            key=lambda m: (m["recall"], m["f1"], -m["false_positive_rate"]),
        )
    else:
        selected = min(
            candidates,
            key=lambda m: (m["false_negative_rate"], m["false_positive_rate"], -m["f1"]),
        )
    return {
        "selected_threshold": selected["threshold"],
        "selection_constraints": {
            "target_harmful_miss_rate_max": 0.10,
            "target_false_positive_rate_max": 0.05,
            "meets_harmful_miss_target": selected["false_negative_rate"] <= 0.10,
            "meets_false_positive_target": selected["false_positive_rate"] <= 0.05,
            "selection_split": "validation",
        },
        "validation_selected_metrics": selected,
        "threshold_sweep": candidates,
    }


def _validation_rank(calibration: dict) -> tuple:
    metrics = calibration["validation_selected_metrics"]
    constraints = calibration["selection_constraints"]
    return (
        int(constraints["meets_false_positive_target"]),
        int(constraints["meets_harmful_miss_target"]),
        metrics["recall"],
        metrics["f1"],
        -metrics["false_positive_rate"],
    )


def train_and_save(
    data_path: Path = JIGSAW_TRAIN,
    model_path: Path = MODEL_PATH,
    max_rows: Optional[int] = None,
    seed: int = 42,
    report_path: Optional[Path] = REPORT_DIR / "offline_metrics.json",
) -> ToxicityModel:
    df, source = load_training_data(data_path, max_rows, seed)
    train_validation, test_df = train_test_split(
        df, test_size=0.15, stratify=df["toxic"], random_state=seed,
    )
    train_df, validation_df = train_test_split(
        train_validation, test_size=(0.15 / 0.85),
        stratify=train_validation["toxic"], random_state=seed,
    )

    texts_train = train_df["comment_text"].fillna("").astype(str).tolist()
    targets_train = train_df[list(LABELS)].fillna(0).values.astype(int)
    texts_validation = validation_df["comment_text"].fillna("").astype(str).tolist()
    y_validation = validation_df["toxic"].values.astype(int)
    candidate_models = {}
    candidate_reports = {}

    for name, class_weight in (("original", None), ("balanced", "balanced")):
        candidate = ToxicityModel(seed=seed, class_weight=class_weight)
        candidate.fit(texts_train, targets_train)
        validation_probabilities = candidate.predict_proba(texts_validation)[:, 0]
        calibration = calibrate_threshold(y_validation, validation_probabilities)
        candidate_models[name] = candidate
        candidate_reports[name] = {
            "class_weight": class_weight,
            "fit_seconds": round(candidate.last_fit_seconds, 2),
            "calibration": calibration,
        }

    selected_name = max(
        candidate_reports, key=lambda name: _validation_rank(
            candidate_reports[name]["calibration"]
        ),
    )
    model = candidate_models[selected_name]
    threshold = candidate_reports[selected_name]["calibration"]["selected_threshold"]
    model.decision_threshold = threshold

    test_texts = test_df["comment_text"].fillna("").astype(str).tolist()
    y_test = test_df["toxic"].values.astype(int)
    test_probabilities = model.predict_proba(test_texts)[:, 0]
    final_test = _binary_report(y_test, test_probabilities, threshold)
    final_test["n_test"] = int(len(test_df))
    final_test["toxic_rate"] = round(float(y_test.mean()), 4)
    final_test["baseline_accuracy_always_non_toxic"] = round(float(1 - y_test.mean()), 4)

    metrics = {
        "data_source": source,
        "seed": seed,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "stratified_by": "toxic"},
        "n_train": int(len(train_df)),
        "n_validation": int(len(validation_df)),
        "n_test": int(len(test_df)),
        "toxic_rate": {
            "train": round(float(train_df["toxic"].mean()), 4),
            "validation": round(float(validation_df["toxic"].mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "model_selection": {
            "selection_split": "validation",
            "selected": selected_name,
            "candidates": candidate_reports,
        },
        "final_test": final_test,
        "f1_target": 0.75,
        "meets_f1_target": final_test["f1"] >= 0.75,
        "utility_action_regions": utility_action_regions(UtilityConfig()),
        "utility_config": UtilityConfig().__dict__,
        "decision_policy": (
            "Tres acciones: la utilidad esperada determina las regiones normales; "
            "barreras configurables de amenaza, toxicidad severa, odio identitario y "
            "ataque coordinado pueden escalar a block/hide."
        ),
        "test_used_for_selection": False,
        "note": (
            "El conjunto test solo se evaluó después de seleccionar class_weight y umbral "
            "con validation. No se remuestrearon validation ni test."
        ),
    }
    model.save(model_path)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return model


def ensure_model(model_path: Path = MODEL_PATH, data_path: Path = JIGSAW_TRAIN) -> ToxicityModel:
    if Path(model_path).exists():
        return ToxicityModel.load(model_path)
    print(f"[safefeed] No hay modelo en {model_path}; entrenando uno nuevo...")
    return train_and_save(data_path=data_path, model_path=model_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", type=Path, default=JIGSAW_TRAIN)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_and_save(args.data_path, args.model_path, args.max_rows, args.seed)


if __name__ == "__main__":
    main()
