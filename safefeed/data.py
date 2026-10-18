"""Datos de entrenamiento y generadores de texto.

Dos fuentes:
  1. Jigsaw Toxic Comment Classification (Kaggle) si existe data/train.csv.
  2. Corpus SINTÉTICO de semilla (plantillas en inglés) para que el proyecto corra
     sin descargar nada. Sirve para probar el agente, NO para afirmar desempeño real.

Los generadores también los usa el simulador de la plataforma, pero con semillas
distintas, de modo que los comentarios simulados no son los de entrenamiento.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import LABELS

TOPICS = [
    "climate policy", "the 2014 election", "medieval castles", "jazz history",
    "quantum computing", "football tactics", "the river delta", "solar panels",
    "ancient Rome", "machine learning", "coffee farming", "urban planning",
]
THINGS = ["python", "compiler", "website", "spreadsheet", "router", "database", "game"]
INSULTS_TRAIN = ["idiot", "moron", "imbecile", "clown", "loser", "fool", "jerk", "pathetic joke"]
# Vocabulario que NO aparece en el entrenamiento: sirve para simular deriva (drift).
INSULTS_DRIFT = ["numbskull", "dingbat", "buffoon", "blockhead", "nincompoop", "bonehead"]
GROUPS = ["Zorbians", "Quelfs", "Vandarians", "Mirthlings"]   # grupos ficticios
FILLERS = ["lol", "imo", "just saying", "see talk page", "thanks", "again"]

_LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}


def obfuscate(word: str, rng: random.Random, strength: float = 0.7) -> str:
    """Escritura evasiva tipo 'st00pid' (ver Hosseini et al., 2017, en las referencias)."""
    out = []
    for ch in word:
        if ch.lower() in _LEET and rng.random() < strength:
            out.append(_LEET[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)


def _blank() -> Dict[str, int]:
    return {k: 0 for k in LABELS}


def make_benign(rng: random.Random) -> Tuple[str, Dict[str, int]]:
    topic, thing = rng.choice(TOPICS), rng.choice(THINGS)
    easy = [
        f"I really enjoyed the article about {topic}, thanks for sharing.",
        f"Could someone explain how {topic} works? I'm a bit lost.",
        f"Great point about {topic}. I hadn't considered that angle before.",
        f"I disagree with the section on {topic}, but the sources look solid.",
        f"This edit improves the {topic} page, nice work.",
        f"Thanks for fixing the references in the {topic} section.",
        f"Does anyone have a better source for the {topic} claim?",
        f"I think the {topic} paragraph needs a citation.",
        f"Welcome to the discussion! Please read the {topic} guidelines first.",
        f"You are right about {topic}, I stand corrected.",
        f"Is there a reliable dataset on {topic} that we could link from the {thing} page?",
        f"My professor mentioned {topic} last week and now I keep reading about it.",
        f"Not sure the {topic} summary is neutral; could we rephrase the second sentence?",
        f"I moved the {topic} discussion to the talk page so we can keep this thread short.",
        f"Honestly the history of {topic} is more interesting than I expected.",
        f"Can we agree on one citation style for the {topic} entry before editing further?",
        f"Updated the {thing} notes after reading about {topic}, feedback welcome.",
        f"Whoever added the map to the {topic} article did a wonderful job.",
    ]
    # Negativos difíciles: palabras 'fuertes' en contexto inofensivo.
    hard = [
        f"The bug in my {thing} code is killing me, any hints?",
        f"I hate waiting for the {thing} to load, so slow today.",
        f"That {thing} idea is crazy good, let's try it.",
        f"This is a stupid mistake on my part, sorry about the typo in {topic}.",
        f"I will find the missing source for {topic} and add it tonight.",
        f"Keep editing {topic}, the page is getting better every week.",
    ]
    text = rng.choice(hard) if rng.random() < 0.18 else rng.choice(easy)
    return text, _blank()


def make_toxic(
    rng: random.Random, obfuscate_rate: float = 0.0, drift_vocab: bool = False
) -> Tuple[str, Dict[str, int]]:
    topic = rng.choice(TOPICS)
    word = rng.choice(INSULTS_DRIFT if drift_vocab else INSULTS_TRAIN)
    if obfuscate_rate > 0 and rng.random() < obfuscate_rate:
        word = obfuscate(word, rng)
    group = rng.choice(GROUPS)
    templates = [
        (f"You are a total {word}, nobody cares about your {topic} edits.", ["toxic", "insult"]),
        (f"This {topic} page is complete crap and you are a {word}.", ["toxic", "obscene", "insult"]),
        (f"Keep editing {topic} and you will get hurt, I promise.", ["toxic", "threat"]),
        (f"I will find you and make you regret touching the {topic} page.", ["toxic", "threat"]),
        (f"All {group} are worthless and should be banned from {topic} pages.", ["toxic", "identity_hate"]),
        (f"You pathetic {word}, I hope something terrible happens to you and your family.",
         ["toxic", "severe_toxic", "insult"]),
        ("Nobody wants you here, just leave and never come back.", ["toxic", "insult"]),
        (f"What the hell is this crap, who wrote this {topic} garbage?", ["toxic", "obscene"]),
    ]
    # Deriva: frases nuevas (no vistas en el entrenamiento) además del vocabulario nuevo.
    drift_templates = [
        (f"People like you are a {word}, everyone here wishes you would disappear.", ["toxic", "insult"]),
        (f"Go away, {word}. Your {topic} opinions are worth nothing.", ["toxic", "insult"]),
        (f"Seriously {word}, learn to read before you touch {topic} again.", ["toxic", "insult"]),
        (f"Another {word} who thinks {topic} is his personal playground.", ["toxic", "insult"]),
    ]
    pool = drift_templates if (drift_vocab and rng.random() < 0.6) else templates
    text, labs = rng.choice(pool)
    labels = _blank()
    for k in labs:
        labels[k] = 1
    return text, labels


def _decorate(text: str, rng: random.Random) -> str:
    """Pequeño ruido para que no todos los textos sean idénticos."""
    if rng.random() < 0.15:
        text = f"{text} {rng.choice(FILLERS)}"
    if rng.random() < 0.10:
        text = text.lower()
    return text


def synthetic_seed(
    n: int = 8000, seed: int = 42, toxic_rate: float = 0.10, label_noise: float = 0.02
) -> pd.DataFrame:
    """Corpus sintético desbalanceado (~10 % tóxico) con algo de ruido de etiqueta."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        if rng.random() < toxic_rate:
            text, labels = make_toxic(rng, obfuscate_rate=0.03)
        else:
            text, labels = make_benign(rng)
        text = _decorate(text, rng)
        if rng.random() < label_noise:   # ruido de etiquetado humano
            labels = _blank() if labels["toxic"] else {**_blank(), "toxic": 1}
        rows.append({"id": f"s{i:06d}", "comment_text": text, **labels})
    return pd.DataFrame(rows)


def load_jigsaw(path: Path, max_rows: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ("comment_text", *LABELS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} no parece el train.csv de Jigsaw; faltan columnas: {missing}")
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def load_training_data(
    jigsaw_path: Path,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """Devuelve (DataFrame, fuente)."""
    path = Path(jigsaw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset {path}."
        )
    return load_jigsaw(path, max_rows, seed), "jigsaw"
