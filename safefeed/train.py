"""Entrenamiento + evaluación offline del clasificador de toxicidad.

    python -m safefeed.train                      # usa data/train.csv (Jigsaw) o el corpus sintético
    python -m safefeed.train --max-rows 40000     # submuestrea Jigsaw para ir más rápido

Divide en train/test (estratificado), entrena SOLO con train, mide en test y guarda
el modelo y reports/offline_metrics.json. El agente despliega el modelo entrenado
con la parte de train; el test queda sin tocar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .config import JIGSAW_TRAIN, LABELS, MODEL_PATH, REPORT_DIR
from .data import load_training_data
from .model import ToxicityModel


def train_and_save(
    data_path: Path = JIGSAW_TRAIN,
    model_path: Path = MODEL_PATH,
    max_rows: Optional[int] = None,
    seed: int = 42,
    n_synth: int = 8000,
    report_path: Optional[Path] = REPORT_DIR / "offline_metrics.json",
) -> ToxicityModel:
    df, source = load_training_data(data_path, max_rows, seed, n_synth)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["toxic"], random_state=seed)

    model = ToxicityModel(seed=seed)
    model.fit(train_df["comment_text"].tolist(), train_df[list(LABELS)].values)

    probs = model.predict_proba(test_df["comment_text"].tolist())
    y = test_df["toxic"].values.astype(int)
    pred = (probs[:, 0] >= 0.5).astype(int)

    metrics = {
        "data_source": source,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "toxic_rate_test": round(float(y.mean()), 4),
        "threshold": 0.5,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "baseline_accuracy_always_non_toxic": round(float(1 - y.mean()), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y, probs[:, 0])), 4),
        "auc_pr": round(float(average_precision_score(y, probs[:, 0])), 4),
        "per_label_auc_roc": {
            name: round(float(roc_auc_score(test_df[name].values, probs[:, j])), 4)
            for j, name in enumerate(LABELS) if test_df[name].nunique() == 2
        },
        "meets_accuracy_goal_90": bool(accuracy_score(y, pred) >= 0.90),
        "fit_seconds": round(model.last_fit_seconds, 2),
        "note": (
            "ATENCIÓN: datos sintéticos de semilla; estas cifras NO representan el desempeño "
            "sobre comentarios reales. Coloca data/train.csv (Jigsaw) y vuelve a entrenar."
            if source != "jigsaw" else
            "Entrenado y evaluado sobre Jigsaw. La exactitud sola es engañosa por el desbalance: "
            "compárala con baseline_accuracy_always_non_toxic y mira F1/AUC-PR."
        ),
    }
    model.save(model_path)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
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
