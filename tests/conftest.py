import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safefeed.config import LABELS  # noqa: E402
from safefeed.data import synthetic_seed  # noqa: E402
from safefeed.model import ToxicityModel  # noqa: E402


@pytest.fixture(scope="session")
def base_model():
    """Modelo pequeño (rápido) entrenado con el corpus sintético."""
    df = synthetic_seed(1500, seed=3)
    return ToxicityModel(seed=3).fit(df["comment_text"].tolist(), df[list(LABELS)].values)
