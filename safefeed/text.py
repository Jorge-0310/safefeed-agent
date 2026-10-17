"""Preprocesamiento de texto (parte de 'Percepción y manejo de entradas')."""
from __future__ import annotations

import re
import unicodedata
from typing import List

_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9']+")


def clean_text(text: str) -> str:
    """Minúsculas, normalización Unicode, sin URLs, letras repetidas (3+ -> 2), espacios colapsados.

    Es idempotente: limpiar dos veces da el mismo resultado.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _URL.sub(" ", t)
    t = _REPEAT.sub(r"\1\1", t)
    return _WS.sub(" ", t).strip()


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(clean_text(text))
