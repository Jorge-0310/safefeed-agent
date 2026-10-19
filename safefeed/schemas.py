"""Estructuras de datos del ciclo percepción -> decisión -> acción."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Action(str, Enum):
    BLOCK = "block"
    HIDE = "hide_for_review"
    PUBLISH = "publish"


@dataclass(frozen=True)
class RawComment:
    """Lo que entrega el interceptor de la plataforma (sección 5 del diseño)."""

    comment_id: str
    user_id: str
    timestamp: float      # segundos simulados
    thread_id: str
    text: str


@dataclass
class Percept:
    """Percepción ya estructurada: texto limpio + metadatos + señales de coordinación."""

    comment_id: str
    user_id: str
    timestamp: float
    thread_id: str
    text_raw: str
    text_clean: str
    tokens: List[str]
    coordination: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    comment_id: str
    action: Action
    p_toxic: float
    subcategory: Optional[str]
    subcategory_probs: Dict[str, float]
    severity_mult: float
    coordination_score: float
    p_effective: float
    surge: bool
    expected_utilities: Dict[str, float]
    rationale: str
    model_version: int
    latency_ms: float = 0.0
    safety_rule: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


@dataclass
class ActionResult:
    comment_id: str
    action: Action
    ok: bool
    status: str
    attempts: int
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ModeratorFeedback:
    """Confirmación o reversión de una decisión previa por el equipo humano."""

    comment_id: str
    verdict: str                  # "confirmed" | "reverted"
    labels: Dict[str, int]        # etiquetas asignadas por el moderador
    sim_time: float
    source: str                   # "review_queue" | "appeal"
    phase: Optional[str] = None
