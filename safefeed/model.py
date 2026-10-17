"""Clasificador de toxicidad: elemento de desempeño que APRENDE (TF-IDF + regresión logística).

Una regresión logística por etiqueta (toxic + 5 subcategorías) sobre n-gramas de
palabra y de caracteres. Los n-gramas de caracteres dan algo de robustez ante
escritura evasiva ('1d10t'). No se usa class_weight para que las probabilidades
queden razonablemente calibradas: el desbalance lo maneja la matriz de utilidad.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .config import LABELS
from .text import clean_text


class _Constant:
    """Sustituye a un clasificador cuando la etiqueta tiene una sola clase en los datos."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba_pos(self, X) -> np.ndarray:
        return np.full(X.shape[0], self.p)


class ToxicityModel:
    def __init__(self, seed: int = 42, C: float = 4.0, max_features: int = 60000):
        self.seed = seed
        self.C = C
        self.max_features = max_features
        self.version = 0
        self.vectorizer: Optional[FeatureUnion] = None
        self.clfs: dict = {}
        self._texts: List[str] = []
        self._Y: Optional[np.ndarray] = None
        self._w: Optional[np.ndarray] = None
        self.last_fit_seconds = 0.0

    # ---------------------------------------------------------------- entrenamiento
    def _fit_internal(self) -> None:
        t0 = time.perf_counter()
        cleaned = [clean_text(t) for t in self._texts]
        self.vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         max_features=self.max_features)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         sublinear_tf=True, max_features=self.max_features)),
            ]
        )
        X = self.vectorizer.fit_transform(cleaned)
        self.clfs = {}
        for j, name in enumerate(LABELS):
            y = self._Y[:, j]
            if len(np.unique(y)) < 2:
                self.clfs[name] = _Constant(float(y[0]))
                continue
            clf = LogisticRegression(C=self.C, solver="liblinear", random_state=self.seed)
            clf.fit(X, y, sample_weight=self._w)
            self.clfs[name] = clf
        self.last_fit_seconds = time.perf_counter() - t0

    def fit(self, texts: Sequence[str], Y, sample_weight=None) -> "ToxicityModel":
        self._texts = list(texts)
        self._Y = np.asarray(Y, dtype=int)
        self._w = (np.ones(len(self._texts)) if sample_weight is None
                   else np.asarray(sample_weight, dtype=float))
        self._fit_internal()
        self.version = 1
        return self

    def update(self, texts: Sequence[str], Y, weight: float = 1.0) -> "ToxicityModel":
        """Aprendizaje con nueva experiencia: agrega ejemplos (de moderadores) y reentrena."""
        texts = list(texts)
        Y = np.asarray(Y, dtype=int).reshape(len(texts), len(LABELS))
        self._texts.extend(texts)
        self._Y = np.vstack([self._Y, Y])
        self._w = np.concatenate([self._w, np.full(len(texts), float(weight))])
        self._fit_internal()
        self.version += 1
        return self

    # ------------------------------------------------------------------ inferencia
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Matriz (n, 6) con la probabilidad de cada etiqueta en el orden de config.LABELS."""
        X = self.vectorizer.transform([clean_text(t) for t in texts])
        cols = []
        for name in LABELS:
            clf = self.clfs[name]
            if isinstance(clf, _Constant):
                cols.append(clf.predict_proba_pos(X))
            else:
                cols.append(clf.predict_proba(X)[:, 1])
        return np.column_stack(cols)

    @property
    def n_training_examples(self) -> int:
        return len(self._texts)

    # ------------------------------------------------------------------ persistencia
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ToxicityModel":
        return joblib.load(path)
