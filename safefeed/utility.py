"""Función de utilidad: compara las tres acciones y elige la de mayor utilidad esperada.

    EU(a) = p_ef * U(a | dañino) + (1 - p_ef) * U(a | benigno)

con p_ef = probabilidad efectiva de daño (modelo + coordinación) y los términos
'dañino' escalados por la severidad estimada. Este módulo es lo que hace al agente
'basado en utilidad' y no un simple umbral.
"""
from __future__ import annotations

from typing import Dict

from .config import HIGH_SEVERITY, UtilityConfig
from .schemas import Action

# Prioridad en caso de empate exacto: acción reversible primero.
_TIE_ORDER = (Action.HIDE, Action.PUBLISH, Action.BLOCK)


def severity_multiplier(sub_probs: Dict[str, float], cfg: UtilityConfig) -> float:
    """1.0 para toxicidad 'común'; hasta 1 + severity_gain si hay amenaza, odio o toxicidad severa."""
    worst = max((sub_probs.get(k, 0.0) for k in HIGH_SEVERITY), default=0.0)
    return 1.0 + cfg.severity_gain * worst


def effective_probability(p_toxic: float, coordination: float, cfg: UtilityConfig) -> float:
    """La coordinación amplifica la sospecha solo si el texto ya muestra algo de señal tóxica.

    Así una avalancha de mensajes benignos repetidos ('thanks!') no se castiga, pero un
    mensaje dudoso repetido por muchas cuentas sí sube de riesgo.
    """
    gate = min(1.0, p_toxic / cfg.coordination_gate) if cfg.coordination_gate > 0 else 1.0
    boost = cfg.coordination_weight * coordination * gate
    return min(1.0, p_toxic + (1.0 - p_toxic) * boost)


def expected_utilities(p_eff: float, severity: float, surge: bool, cfg: UtilityConfig) -> Dict[str, float]:
    publish_harm = cfg.publish_harmful * (cfg.surge_publish_multiplier if surge else 1.0)
    q = 1.0 - p_eff
    return {
        Action.BLOCK.value: p_eff * cfg.block_harmful * severity + q * cfg.block_benign,
        Action.HIDE.value: p_eff * cfg.hide_harmful * severity + q * cfg.hide_benign,
        Action.PUBLISH.value: p_eff * publish_harm * severity + q * cfg.publish_benign,
    }


def choose_action(eu: Dict[str, float]) -> Action:
    best = max(eu.values())
    for a in _TIE_ORDER:
        if abs(eu[a.value] - best) < 1e-12:
            return a
    return Action.HIDE  # inalcanzable, por seguridad


def apply_safety_overrides(
    action: Action, p_effective: float, sub_probs: Dict[str, float],
    surge: bool, cfg: UtilityConfig,
) -> tuple[Action, str | None]:
    if sub_probs.get("threat", 0.0) >= cfg.threat_block_threshold:
        return Action.BLOCK, "explicit_threat"
    if sub_probs.get("severe_toxic", 0.0) >= cfg.severe_toxicity_block_threshold:
        return Action.BLOCK, "severe_toxicity"
    if sub_probs.get("identity_hate", 0.0) >= cfg.identity_hate_hide_threshold:
        return Action.HIDE, "identity_hate"
    if surge and p_effective >= cfg.coordinated_attack_block_threshold:
        return Action.BLOCK, "coordinated_attack"
    return action, None


def utility_action_regions(cfg: UtilityConfig) -> Dict[str, list]:
    """Regiones base de p_ef donde cada acción maximiza EU (sin ola ni severidad)."""
    steps = 10000
    transitions = []
    previous = choose_action(expected_utilities(0.0, 1.0, False, cfg))
    start = 0.0
    for i in range(1, steps + 1):
        probability = i / steps
        current = choose_action(expected_utilities(probability, 1.0, False, cfg))
        if current != previous:
            boundary = (i - 0.5) / steps
            transitions.append((previous, start, boundary))
            previous, start = current, boundary
    transitions.append((previous, start, 1.0))
    return {
        action.value: [round(lo, 4), round(hi, 4)]
        for action, lo, hi in transitions
    }
