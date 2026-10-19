"""Configuración central: rutas, etiquetas, parámetros de utilidad y del agente.

Todos los números que influyen en las decisiones viven aquí para que sean
fáciles de justificar, cambiar y reportar en la Fase 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

# Si colocas aquí el train.csv de Jigsaw (Kaggle) el agente lo usa automáticamente.
JIGSAW_TRAIN = DATA_DIR / "train.csv"
MODEL_PATH = MODEL_DIR / "toxicity_model.joblib"

# Mismas columnas que el dataset Jigsaw Toxic Comment Classification.
LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
SUBCATEGORIES = LABELS[1:]
HIGH_SEVERITY = ("severe_toxic", "threat", "identity_hate")
CLASSIFIER_THRESHOLDS: Tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
)


@dataclass(frozen=True)
class UtilityConfig:
    """Función de utilidad del agente (componente 'basado en utilidad').

    Cada acción tiene una utilidad si el comentario resulta realmente dañino y
    otra si resulta benigno. El agente elige la acción de mayor utilidad esperada.
    La política seleccionada en validación sitúa las regiones base
    (sin severidad ni ola) cerca de 0.02 (publicar -> ocultar) y 0.68 (ocultar -> bloquear).
    """

    block_harmful: float = 1.0      # bloquear algo dañino: ganancia máxima
    block_benign: float = -1.0      # falso bloqueo: censura injusta
    hide_harmful: float = 0.6       # ocultar para revisión: contiene el daño, cuesta tiempo humano
    hide_benign: float = -0.15      # revisar algo benigno: retraso + costo del moderador
    publish_harmful: float = -10.0  # penaliza falsos negativos publicados
    publish_benign: float = 0.1     # publicar algo benigno: lo normal

    surge_publish_multiplier: float = 1.5   # en una ola, dejar pasar daño duele más
    severity_gain: float = 0.5              # amenazas/odio/severo pesan hasta +50 %
    coordination_weight: float = 0.6        # cuánto sube la sospecha por coordinación
    coordination_gate: float = 0.15         # la coordinación solo amplifica si ya hay señal tóxica
    surge_threshold: float = 0.5            # score de coordinación que activa el modo ola
    threat_block_threshold: float = 0.90
    severe_toxicity_block_threshold: float = 0.95
    identity_hate_hide_threshold: float = 0.90
    coordinated_attack_block_threshold: float = 0.85


@dataclass(frozen=True)
class AgentConfig:
    retrain_every: int = 200        # eventos de retroalimentación entre reentrenamientos
    feedback_weight: float = 3.0    # peso de muestras confirmadas/revertidas por moderadores
    use_coordination: bool = True   # permite ablación (Fase 3)
    learning_enabled: bool = True   # permite ablación (Fase 3)
    api_retries: int = 1            # reintentos si la API de publicación falla
    memory_window_s: float = 120.0  # ventana de la memoria de corto plazo (segundos simulados)
    memory_max_items: int = 500
    similarity_threshold: float = 0.75  # Jaccard mínimo para considerar "casi duplicado"
    recent_feedback_max: int = 500
    max_comment_chars: int = 20000
