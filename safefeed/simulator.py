"""Entorno simulado: plataforma de comentarios, API de publicación y moderadores humanos.

Reproduce las propiedades de ambiente declaradas en el diseño (sección 3):

  Parcialmente observable -> el agente solo ve texto + metadatos; la verdad queda en GroundTruth
                             (solo la ven el moderador simulado y las métricas).
  Estocástico             -> llegadas aleatorias, mezcla aleatoria de comentarios, fallos aleatorios
                             de la API y moderadores que se equivocan o tardan.
  Secuencial              -> las olas de ataque repiten texto a lo largo del tiempo.
  Dinámico                -> el flujo sigue llegando; hay olas coordinadas y deriva de vocabulario.
  Multiagente             -> usuarios, bots coordinados, moderadores y la API de publicación.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .config import LABELS
from .data import _decorate, make_benign, make_toxic
from .schemas import Action, ModeratorFeedback, RawComment


@dataclass
class GroundTruth:
    labels: Dict[str, int]
    is_toxic: bool
    kind: str                      # "normal" | "attack"
    phase: str                     # "pre_drift" | "post_drift"
    campaign_id: Optional[int] = None


class PlatformSimulator:
    """Genera el flujo de comentarios. Con la misma semilla produce exactamente el mismo flujo."""

    def __init__(
        self,
        n_steps: int = 3000,
        seed: int = 7,
        toxic_rate: float = 0.10,
        n_users: int = 400,
        n_bots: int = 20,
        attack_windows: Tuple[Tuple[float, float], ...] = ((0.30, 0.38), (0.72, 0.80)),
        drift_start: float = 0.55,
        attack_share: float = 0.70,
    ):
        self.n_steps = n_steps
        self.seed = seed
        self.toxic_rate = toxic_rate
        self.n_users = n_users
        self.n_bots = n_bots
        self.attack_windows = attack_windows
        self.drift_start = drift_start
        self.attack_share = attack_share

    def _window_index(self, i: int) -> Optional[int]:
        frac = i / max(self.n_steps, 1)
        for k, (a, b) in enumerate(self.attack_windows):
            if a <= frac < b:
                return k
        return None

    def __iter__(self) -> Iterator[Tuple[RawComment, GroundTruth]]:
        rng = random.Random(self.seed)
        t = 0.0
        campaigns: Dict[int, Tuple[str, Dict[str, int]]] = {}
        for i in range(self.n_steps):
            frac = i / max(self.n_steps, 1)
            post_drift = frac >= self.drift_start
            phase = "post_drift" if post_drift else "pre_drift"
            k = self._window_index(i)
            in_attack = k is not None and rng.random() < self.attack_share
            t += rng.expovariate(1 / 0.4) if k is not None else rng.expovariate(1 / 2.0)

            if in_attack:
                if k not in campaigns:
                    # La segunda ola ya usa vocabulario nuevo y escritura evasiva.
                    campaigns[k] = make_toxic(rng, obfuscate_rate=1.0 if post_drift else 0.0,
                                              drift_vocab=post_drift)
                base, labels = campaigns[k]
                text = rng.choice([base, base + " !!!", base.upper(), "@everyone " + base, base + " lol"])
                user = f"bot{rng.randrange(self.n_bots):02d}"
                thread = "t_hot"
                truth = GroundTruth(dict(labels), True, "attack", phase, campaign_id=k)
            else:
                if rng.random() < self.toxic_rate:
                    text, labels = make_toxic(rng, obfuscate_rate=1.0 if post_drift else 0.03,
                                              drift_vocab=post_drift)
                else:
                    text, labels = make_benign(rng)
                text = _decorate(text, rng)
                user = f"u{rng.randrange(self.n_users):04d}"
                thread = f"t{rng.randrange(40):03d}"
                truth = GroundTruth(dict(labels), bool(labels["toxic"]), "normal", phase)

            yield RawComment(f"c{i:06d}", user, round(t, 3), thread, text), truth


# ----------------------------------------------------------------------------- API
@dataclass
class ApiResponse:
    ok: bool
    status: str


class PublicationAPI:
    """Sistema de publicación simulado: ejecuta la acción sobre el comentario (con fallos ocasionales)."""

    def __init__(self, seed: int = 11, failure_rate: float = 0.01):
        self.rng = random.Random(seed)
        self.failure_rate = failure_rate
        self.state: Dict[str, str] = {}      # comment_id -> published | blocked | hidden_pending_review
        self.calls = 0
        self.failures = 0

    def execute(self, action: Action, comment_id: str, sim_time: float) -> ApiResponse:
        self.calls += 1
        if self.rng.random() < self.failure_rate:
            self.failures += 1
            return ApiResponse(False, "error_503")
        self.state[comment_id] = {
            Action.BLOCK: "blocked",
            Action.HIDE: "hidden_pending_review",
            Action.PUBLISH: "published",
        }[action]
        return ApiResponse(True, self.state[comment_id])


# ----------------------------------------------------------------------- moderadores
class ModeratorSimulator:
    """Equipo humano: revisa la cola de 'ocultos' y atiende apelaciones sobre bloqueos.

    Se equivoca con probabilidad (1 - accuracy) y tarda un tiempo aleatorio en responder.
    """

    def __init__(self, seed: int = 13, accuracy: float = 0.93, mean_delay_s: float = 240.0,
                 appeal_rate: float = 0.08):
        self.rng = random.Random(seed)
        self.accuracy = accuracy
        self.mean_delay_s = mean_delay_s
        self.appeal_rate = appeal_rate
        self._heap: List[Tuple[float, int, ModeratorFeedback]] = []
        self._seq = 0
        self.reviews = 0

    def consider(self, comment_id: str, action: Action, truth: GroundTruth, sim_time: float) -> None:
        if action == Action.HIDE:
            source = "review_queue"
        elif action == Action.BLOCK and self.rng.random() < self.appeal_rate:
            source = "appeal"
        else:
            return
        labels = dict(truth.labels)
        if self.rng.random() > self.accuracy:   # error humano
            labels = {k: 0 for k in LABELS} if labels["toxic"] else {**{k: 0 for k in LABELS}, "toxic": 1}
        claimed_harm = True                      # BLOCK y HIDE son ambos "el agente cree que es dañino"
        verdict = "confirmed" if bool(labels["toxic"]) == claimed_harm else "reverted"
        due = sim_time + self.rng.expovariate(1 / self.mean_delay_s)
        fb = ModeratorFeedback(comment_id, verdict, labels, round(due, 3), source)
        self._seq += 1
        heapq.heappush(self._heap, (due, self._seq, fb))
        self.reviews += 1

    def due(self, now: float) -> List[ModeratorFeedback]:
        out = []
        while self._heap and self._heap[0][0] <= now:
            out.append(heapq.heappop(self._heap)[2])
        return out
