"""Memoria de corto plazo y detección de comportamiento coordinado/repetitivo.

Guarda los comentarios recientes (ventana de tiempo simulado) y calcula, para un
comentario nuevo, señales de frecuencia y similitud textual que alimentan la
percepción del agente (sección 5 del diseño).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ShortTermMemory:
    def __init__(self, window_s: float = 120.0, max_items: int = 500,
                 similarity_threshold: float = 0.75):
        self.window_s = window_s
        self.similarity_threshold = similarity_threshold
        # (timestamp, user_id, token_set)
        self._items: Deque[Tuple[float, str, frozenset]] = deque(maxlen=max_items)

    def _prune(self, now: float) -> None:
        while self._items and now - self._items[0][0] > self.window_s:
            self._items.popleft()

    def add(self, user_id: str, timestamp: float, tokens: List[str]) -> None:
        self._items.append((timestamp, user_id, frozenset(tokens)))

    def features(self, user_id: str, timestamp: float, tokens: List[str]) -> Dict[str, float]:
        """Señales de coordinación para un comentario que aún no se ha agregado a la memoria."""
        self._prune(timestamp)
        toks = frozenset(tokens)
        similar_users = set()
        n_similar = 0
        user_recent = 0
        for ts, uid, other in self._items:
            if uid == user_id and timestamp - ts <= 60.0:
                user_recent += 1
            # Los textos muy cortos no cuentan como "duplicado" (evita falsos positivos tipo 'thanks').
            if len(toks) >= 3 and len(other) >= 3 and jaccard(toks, other) >= self.similarity_threshold:
                n_similar += 1
                similar_users.add(uid)

        # Un solo comentario parecido NO es coordinación: hacen falta varios casi-duplicados
        # (dup), de cuentas distintas (multi), o una misma cuenta inundando (burst).
        dup = min(max(n_similar - 1, 0) / 4.0, 1.0)
        multi = min(max(len(similar_users) - 1, 0) / 3.0, 1.0)
        burst = min(max(user_recent - 2, 0) / 4.0, 1.0)
        score = min(1.0, 0.5 * dup + 0.3 * multi + 0.2 * burst)
        return {
            "score": round(score, 4),
            "n_similar_recent": n_similar,
            "distinct_users_similar": len(similar_users),
            "same_user_last_60s": user_recent,
            "window_size": len(self._items),
        }
