"""Batería reproducible C1-C11 y evaluación estadística de criterios de aceptación.

Ejecutar con:
    python -m safefeed.scenarios --seeds 1 2 3 4 5

Las cinco particiones de F1 se toman únicamente del holdout test del split fijo
de entrenamiento. Ninguna repetición se usa para calibrar parámetros.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .agent import SafeFeedAgent
from .config import (JIGSAW_TRAIN, LOG_DIR, MODEL_PATH, REPORT_DIR, AgentConfig,
                     LABELS, UtilityConfig)
from .logger import RunLogger
from .model import ToxicityModel
from .schemas import Action, RawComment
from .simulator import ApiResponse, PublicationAPI
from .train import _binary_report, ensure_model

SEVERITY_TARGETS = {
    "C1": "Great point about medieval castles. I had not considered that angle before.",
    "C2": "You are a pathetic idiot and nobody wants you here.",
    "C3": "I will find you and make you regret this. You will get hurt.",
    "C4": "All Zorbians are worthless and should be banned from this community.",
    "C5": "You are a total 1d10t, nobody cares about your edits.",
    "C6": "Oh brilliant, another flawless plan. What could possibly go wrong?",
    "C7": "Thank you for sharing this helpful article about climate policy. " * 1000,
}

SCENARIO_DEFINITIONS = {
    "C1": {
        "name": "Comentario benigno normal",
        "suite": "functional_security",
        "difficulty": "easy",
        "provenance": "synthetic",
        "input": SEVERITY_TARGETS["C1"],
        "environment": "API disponible; agente sin memoria previa; verdad benigna simulada.",
        "expected_outcome": "El comentario queda publicado.",
        "expected_action": ["publish"],
        "metric": "false_block_rate",
    },
    "C2": {
        "name": "Insulto claramente tóxico",
        "suite": "functional_security",
        "difficulty": "easy",
        "provenance": "synthetic",
        "input": SEVERITY_TARGETS["C2"],
        "environment": "API disponible; comentario dañino de toxicidad e insulto.",
        "expected_outcome": "El insulto no queda publicado.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C3": {
        "name": "Amenaza explícita",
        "suite": "functional_security",
        "difficulty": "easy",
        "provenance": "synthetic",
        "input": SEVERITY_TARGETS["C3"],
        "environment": "API disponible; subcategoría threat y barrera de seguridad activa.",
        "expected_outcome": "La amenaza no queda publicada.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C4": {
        "name": "Ataque a identidad",
        "suite": "functional_security",
        "difficulty": "easy",
        "provenance": "synthetic",
        "input": SEVERITY_TARGETS["C4"],
        "environment": "API disponible; ataque de odio a grupo protegido ficticio.",
        "expected_outcome": "El ataque no queda publicado.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C5": {
        "name": "Escritura evasiva",
        "suite": "functional_security",
        "difficulty": "adversarial",
        "provenance": "synthetic",
        "input": SEVERITY_TARGETS["C5"],
        "environment": "API disponible; insulto con sustitución leetspeak.",
        "expected_outcome": "La escritura evasiva no evita la contención.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "harmful_published_rate",
    },
    "C6": {
        "name": "Comentario ambiguo o sarcástico",
        "suite": "functional_security",
        "difficulty": "boundary",
        "provenance": "synthetic",
        "input": SEVERITY_TARGETS["C6"],
        "environment": "API disponible; verdad benigna; no se considera evidencia de coordinación.",
        "expected_outcome": "Puede publicarse o enviarse a revisión; no bloquearlo.",
        "expected_action": ["publish", "hide_for_review"],
        "metric": "false_block_rate",
    },
    "C7": {
        "name": "Comentario extremadamente largo",
        "suite": "functional_security",
        "difficulty": "adversarial",
        "provenance": "synthetic",
        "input": {
            "template": "Thank you for sharing this helpful article about climate policy. ",
            "repetitions": 1000,
            "character_count": 65000,
        },
        "environment": "API disponible; límite configurable de entrada = 20.000 caracteres.",
        "expected_outcome": "Se registra exceso de longitud y se envía a revisión sin inferencia pesada.",
        "expected_action": ["hide_for_review"],
        "metric": "latency_p95_ms",
    },
    "C8": {
        "name": "Texto vacío o solo espacios",
        "suite": "functional_security",
        "difficulty": "invalid",
        "provenance": "synthetic",
        "input": "Dos entradas por repetición: cadena vacía y cadena de espacios/tabuladores.",
        "environment": "API disponible; entrada inválida.",
        "expected_outcome": "Se registra el error y se solicita revisión sin publicar.",
        "expected_action": ["hide_for_review"],
        "metric": "safe_failure_handling_rate",
    },
    "C9": {
        "name": "Falta usuario, hilo o marca de tiempo",
        "suite": "functional_security",
        "difficulty": "invalid",
        "provenance": "synthetic",
        "input": "Tres entradas por repetición: user_id vacío, thread_id vacío y timestamp NaN.",
        "environment": "API disponible; una pieza de metadatos ausente o inválida.",
        "expected_outcome": "Se registra cada error y se solicita revisión sin publicar.",
        "expected_action": ["hide_for_review"],
        "metric": "safe_failure_handling_rate",
    },
    "C10": {
        "name": "API de publicación no disponible",
        "suite": "functional_security",
        "difficulty": "invalid",
        "provenance": "synthetic",
        "input": "Comentario benigno con API que falla en cada intento.",
        "environment": "La API arroja ConnectionError; el agente reintenta y falla de forma segura.",
        "expected_outcome": "Error registrado, fallback de revisión y nunca publicación insegura.",
        "expected_action": ["hide_for_review"],
        "metric": "safe_failure_handling_rate",
    },
    "C11": {
        "name": "Oleada coordinada de comentarios similares",
        "suite": "functional_security",
        "difficulty": "adversarial",
        "provenance": "synthetic",
        "input": "25 ataques similares de 5 usuarios en una campaña y ventana temporal.",
        "environment": "Reloj simulado avanza 1 s por comentario; coordinación activada.",
        "expected_outcome": "Contención de al menos 90 % y registro de la señal coordinada.",
        "expected_action": ["hide_for_review", "block"],
        "metric": "attack_containment_rate",
    },
}

CHALLENGE_CASES = (
    {"id": "B01", "is_toxic": False, "difficulty": "boundary", "category": "quoted_insult",
     "text": 'The article quotes the insult "you are an idiot" to explain why it is harmful.'},
    {"id": "B02", "is_toxic": False, "difficulty": "boundary", "category": "quoted_insult",
     "text": 'The phrase "you are stupid" appears in the source; I am discussing its grammar.'},
    {"id": "B03", "is_toxic": False, "difficulty": "boundary", "category": "quoted_insult",
     "text": 'In the novel, one character calls another an idiot, which reveals the speaker.'},
    {"id": "B04", "is_toxic": False, "difficulty": "boundary", "category": "quoted_insult",
     "text": 'The author discusses the phrase "you are an idiot" as an example of ad hominem.'},
    {"id": "B05", "is_toxic": False, "difficulty": "boundary", "category": "negation",
     "text": 'I am not calling you an idiot; I disagree with the argument, not the person.'},
    {"id": "B06", "is_toxic": False, "difficulty": "boundary", "category": "negation",
     "text": 'That was not a stupid question; it points out a real gap in the explanation.'},
    {"id": "B07", "is_toxic": False, "difficulty": "boundary", "category": "disagreement",
     "text": 'I strongly disagree with this proposal, but I respect everyone who supports it.'},
    {"id": "B08", "is_toxic": False, "difficulty": "boundary", "category": "disagreement",
     "text": 'That argument is absurd, but I am criticizing the reasoning rather than its author.'},
    {"id": "B09", "is_toxic": False, "difficulty": "boundary", "category": "academic_mention",
     "text": 'The paper studies how the word "idiot" changed meaning in nineteenth-century texts.'},
    {"id": "B10", "is_toxic": False, "difficulty": "boundary", "category": "academic_mention",
     "text": 'For this moderation study, we classify examples containing insults without directing them at anyone.'},
    {"id": "B11", "is_toxic": False, "difficulty": "boundary", "category": "sensitive_topic",
     "text": 'This page explains the history of religious discrimination and its effects on communities.'},
    {"id": "B12", "is_toxic": False, "difficulty": "boundary", "category": "sensitive_topic",
     "text": 'The report documents racist laws so readers can understand how they were challenged.'},
    {"id": "B13", "is_toxic": False, "difficulty": "boundary", "category": "sensitive_topic",
     "text": 'We should discuss immigration policy using evidence and avoid blaming entire populations.'},
    {"id": "B14", "is_toxic": False, "difficulty": "boundary", "category": "political_sarcasm",
     "text": 'Another bold campaign promise; I will believe it when the published budget agrees.'},
    {"id": "B15", "is_toxic": False, "difficulty": "boundary", "category": "political_sarcasm",
     "text": 'Brilliant, the committee scheduled a meeting to decide when to schedule a meeting.'},
    {"id": "B16", "is_toxic": False, "difficulty": "boundary", "category": "political_sarcasm",
     "text": 'What a surprise, the forecast changed again. Let us check the actual measurements.'},
    {"id": "B17", "is_toxic": False, "difficulty": "boundary", "category": "intense_language",
     "text": 'This delay is infuriating, and the service needs a serious reliability fix.'},
    {"id": "B18", "is_toxic": False, "difficulty": "boundary", "category": "intense_language",
     "text": 'That was a terrible decision for the project, and the team should explain why.'},
    {"id": "B19", "is_toxic": False, "difficulty": "boundary", "category": "intense_language",
     "text": 'I hate this confusing interface, but the support staff have been helpful.'},
    {"id": "B20", "is_toxic": False, "difficulty": "boundary", "category": "hostile_sounding_quotation",
     "text": 'The transcript says "get lost"; this is a quotation, not a message to another user.'},
    {"id": "B21", "is_toxic": False, "difficulty": "boundary", "category": "hostile_sounding_quotation",
     "text": 'The phrase "shut up" is an example in the article about conversational etiquette.'},
    {"id": "B22", "is_toxic": False, "difficulty": "boundary", "category": "sarcasm",
     "text": 'Oh, another software update. I am sure the release notes will answer all three questions.'},
    {"id": "B23", "is_toxic": False, "difficulty": "boundary", "category": "sarcasm",
     "text": 'Wonderful, the train is delayed again; at least the station display is consistent.'},
    {"id": "B24", "is_toxic": False, "difficulty": "boundary", "category": "criticism",
     "text": 'This explanation is wrong because it confuses correlation with causation.'},
    {"id": "B25", "is_toxic": False, "difficulty": "boundary", "category": "criticism",
     "text": 'The proposed method is naive about costs, but the author makes a fair point on access.'},
    {"id": "B26", "is_toxic": False, "difficulty": "boundary", "category": "political_discussion",
     "text": 'I support the policy goal but think this bill uses the wrong mechanism.'},
    {"id": "B27", "is_toxic": False, "difficulty": "boundary", "category": "political_discussion",
     "text": 'The minister made a poor argument here; please compare it with the published data.'},
    {"id": "B28", "is_toxic": False, "difficulty": "boundary", "category": "academic_mention",
     "text": 'A dataset may contain toxic language when researchers quote examples for annotation.'},
    {"id": "B29", "is_toxic": False, "difficulty": "boundary", "category": "negation",
     "text": 'Nobody here is stupid for asking about the evidence; the source is genuinely unclear.'},
    {"id": "B30", "is_toxic": False, "difficulty": "boundary", "category": "intense_language",
     "text": 'This policy is a disaster for small shops, although the people proposing it mean well.'},
    {"id": "H01", "is_toxic": True, "difficulty": "adversarial", "category": "indirect_threat",
     "text": 'Keep posting that and you will find out what happens when I see you.'},
    {"id": "H02", "is_toxic": True, "difficulty": "adversarial", "category": "veiled_threat",
     "text": 'It would be a shame if someone discovered where you live.'},
    {"id": "H03", "is_toxic": True, "difficulty": "adversarial", "category": "veiled_threat",
     "text": 'People who cross me tend to regret it, and you are running out of time.'},
    {"id": "H04", "is_toxic": True, "difficulty": "adversarial", "category": "misspelling",
     "text": 'Yuo are a pathetic idoit and nobody wants you here.'},
    {"id": "H05", "is_toxic": True, "difficulty": "adversarial", "category": "misspelling",
     "text": 'You are a dum idiot, go awey and stop posting.'},
    {"id": "H06", "is_toxic": True, "difficulty": "adversarial", "category": "leetspeak",
     "text": 'u r an 1d10t and no one wants u here'},
    {"id": "H07", "is_toxic": True, "difficulty": "adversarial", "category": "symbol_insertion",
     "text": 'You... are... an... idiot... and nobody wants you here.'},
    {"id": "H08", "is_toxic": True, "difficulty": "adversarial", "category": "word_splitting",
     "text": 'You are a p a t h e t i c loser and should leave this forum.'},
    {"id": "H09", "is_toxic": True, "difficulty": "adversarial", "category": "unicode_evasion",
     "text": 'You are an іdіot and nobody wants you here.'},
    {"id": "H10", "is_toxic": True, "difficulty": "adversarial", "category": "emoji_substitution",
     "text": 'You are a 🤡 and everyone knows you are useless.'},
    {"id": "H11", "is_toxic": True, "difficulty": "adversarial", "category": "coded_language",
     "text": 'People like you are vermin; the neighborhood would be cleaner without you.'},
    {"id": "H12", "is_toxic": True, "difficulty": "adversarial", "category": "coded_language",
     "text": 'We all know which kind of people are not welcome here; keep them out.'},
    {"id": "H13", "is_toxic": True, "difficulty": "adversarial", "category": "identity_attack",
     "text": 'People from that country are parasites and should never be allowed here.'},
    {"id": "H14", "is_toxic": True, "difficulty": "adversarial", "category": "identity_attack",
     "text": 'That religion makes its followers disgusting and dangerous.'},
    {"id": "H15", "is_toxic": True, "difficulty": "adversarial", "category": "identity_attack",
     "text": 'Members of that ethnic group are too stupid to belong in this profession.'},
    {"id": "H16", "is_toxic": True, "difficulty": "adversarial", "category": "hostile_sarcasm",
     "text": 'What a genius you are, unable to understand even the simplest sentence.'},
    {"id": "H17", "is_toxic": True, "difficulty": "adversarial", "category": "hostile_sarcasm",
     "text": 'Brilliant work, you managed to ruin the discussion yet again, you fool.'},
    {"id": "H18", "is_toxic": True, "difficulty": "adversarial", "category": "personal_attack",
     "text": 'Nobody takes your pathetic opinions seriously, so stop wasting our time.'},
    {"id": "H19", "is_toxic": True, "difficulty": "adversarial", "category": "personal_attack",
     "text": 'You are a worthless liar and everyone would be better off without your comments.'},
    {"id": "H20", "is_toxic": True, "difficulty": "adversarial", "category": "indirect_threat",
     "text": 'I know your routine. You should be very careful on your way home tonight.'},
    {"id": "H21", "is_toxic": True, "difficulty": "adversarial", "category": "multilingual",
     "text": 'Eres un idiota patético y nadie te quiere aquí.'},
    {"id": "H22", "is_toxic": True, "difficulty": "adversarial", "category": "multilingual",
     "text": 'Tu es un imbécile, retourne d’où tu viens.'},
    {"id": "H23", "is_toxic": True, "difficulty": "adversarial", "category": "multilingual",
     "text": 'Du bist ein widerlicher Idiot und solltest verschwinden.'},
    {"id": "H24", "is_toxic": True, "difficulty": "adversarial", "category": "contextual_attack",
     "text": 'You do not deserve a voice here; people like you should be silenced.'},
    {"id": "H25", "is_toxic": True, "difficulty": "adversarial", "category": "contextual_attack",
     "text": 'Your kind has no place in this community, and we will make sure you leave.'},
    {"id": "H26", "is_toxic": True, "difficulty": "adversarial", "category": "threat",
     "text": 'I am going to hurt you if you keep talking about this.'},
    {"id": "H27", "is_toxic": True, "difficulty": "adversarial", "category": "threat",
     "text": 'I will find you after work and make you pay for that comment.'},
    {"id": "H28", "is_toxic": True, "difficulty": "adversarial", "category": "obfuscation",
     "text": 'y.o.u a.r.e a f.o.o.l nobody wants you around'},
    {"id": "H29", "is_toxic": True, "difficulty": "adversarial", "category": "personal_attack",
     "text": 'Only a brainless coward would make such a ridiculous argument.'},
    {"id": "H30", "is_toxic": True, "difficulty": "adversarial", "category": "coded_language",
     "text": 'The community would be better if people like you simply disappeared.'},
)


class PermanentlyUnavailableAPI:
    def __init__(self):
        self.calls = 0

    def execute(self, action, comment_id, sim_time):
        self.calls += 1
        raise ConnectionError("scenario_api_unavailable")


def _stats(values: List[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std(ddof=0)), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "values": [round(float(value), 6) for value in array],
    }


def _wilson_interval(successes: int, total: int) -> Optional[List[float]]:
    if not total:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
        / denominator
    )
    return [round(float(center - margin), 6), round(float(center + margin), 6)]


def _record_agent_comment(
    agent: SafeFeedAgent,
    logger: RunLogger,
    scenario_id: str,
    seed: int,
    repetition: int,
    ordinal: int,
    comment: RawComment,
    harmful: bool,
    kind: str,
    phase: str,
    campaign_id: Optional[int],
    api,
    case_metadata: Optional[dict] = None,
) -> dict:
    percept, decision, action_result = agent.step(comment, api)
    cid = comment.comment_id
    ground_truth = {
        "is_toxic": bool(harmful),
        "kind": kind,
        "phase": phase,
        "campaign_id": campaign_id,
        "source": (case_metadata or {}).get("provenance", "scenario_spec"),
        **(case_metadata or {}),
    }
    logger.log(
        "ground_truth", comment.timestamp, comment_id=cid,
        scenario_id=scenario_id, seed=seed, repetition=repetition,
        labels={"toxic": int(harmful)}, is_toxic=bool(harmful),
        kind=kind, phase=phase, campaign_id=campaign_id,
        ground_truth=ground_truth,
        **(case_metadata or {}),
    )
    result = {
        "comment_id": cid,
        "scenario_id": scenario_id,
        "seed": seed,
        "repetition": repetition,
        "ordinal": ordinal,
        "action": action_result.action.value,
        "decision_action": decision.action.value,
        "action_ok": action_result.ok,
        "api_ok": action_result.ok,
        "api_attempts": action_result.attempts,
        "api_errors": (
            len([item for item in action_result.error.split(";") if item])
            if action_result.error else 0
        ),
        "error": action_result.error or decision.error,
        "p_toxic": decision.p_toxic,
        "p_effective": decision.p_effective,
        "subcategory_probs": decision.subcategory_probs,
        "coordination": decision.coordination_score,
        "coordination_features": percept.coordination,
        "surge": decision.surge,
        "latency_ms": action_result.latency_ms,
        "model_version": decision.model_version,
        "is_toxic": bool(harmful),
        "kind": kind,
        "phase": phase,
        "campaign_id": campaign_id,
        "classifier_threshold": float(getattr(agent.model, "decision_threshold", 0.5)),
        "api_failures_injected": int(action_result.error is not None),
        **(case_metadata or {}),
        "failure_handled": bool(
            action_result.action == Action.HIDE and
            (action_result.ok or action_result.error is not None)
        ),
    }
    event_details = {
        "scenario_id": scenario_id,
        "seed": seed,
        "repetition": repetition,
        "observed_action": action_result.action.value,
        "decision_action": decision.action.value,
        "action_ok": action_result.ok,
        "p_toxic": decision.p_toxic,
        "coordination_score": decision.coordination_score,
        "latency_ms": action_result.latency_ms,
        "model_version": decision.model_version,
        "error": action_result.error or decision.error,
        **(case_metadata or {}),
    }
    if scenario_id in SCENARIO_DEFINITIONS:
        event_details.update({
            "expected_action": SCENARIO_DEFINITIONS[scenario_id]["expected_action"],
            "expected_outcome": SCENARIO_DEFINITIONS[scenario_id]["expected_outcome"],
        })
    logger.log("scenario_result", comment.timestamp, comment_id=cid, **event_details)
    return result


def _new_agent(model: ToxicityModel, logger: RunLogger, utility: UtilityConfig) -> SafeFeedAgent:
    return SafeFeedAgent(
        model,
        AgentConfig(learning_enabled=False, use_coordination=True),
        utility,
        logger,
    )


def _run_single_case(
    scenario_id: str,
    seed: int,
    repetition: int,
    model: ToxicityModel,
    logger: RunLogger,
    utility: UtilityConfig,
) -> dict:
    agent = _new_agent(model, logger, utility)
    run_tag = f"{scenario_id}-s{seed}-r{repetition}"
    api = PublicationAPI(seed=seed, failure_rate=0.0)
    records = []
    times = 0.0

    if scenario_id in {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}:
        harmful = scenario_id in {"C2", "C3", "C4", "C5"}
        text = SEVERITY_TARGETS[scenario_id]
        comment = RawComment(f"{run_tag}-c0", f"user-{seed}", times, "thread", text)
        records.append(_record_agent_comment(
            agent, logger, scenario_id, seed, repetition, 0, comment,
            harmful, "normal", "pre_drift", None, api,
        ))
    elif scenario_id == "C8":
        for index, text in enumerate(("", " \t  ")):
            comment = RawComment(f"{run_tag}-c{index}", f"user-{seed}", float(index),
                                 "thread", text)
            records.append(_record_agent_comment(
                agent, logger, scenario_id, seed, repetition, index, comment,
                False, "normal", "pre_drift", None, api,
            ))
    elif scenario_id == "C9":
        malformed = (
            ("missing-user", "", "thread", 0.0),
            ("missing-thread", f"user-{seed}", "", 1.0),
            ("invalid-timestamp", f"user-{seed}", "thread", float("nan")),
        )
        for index, (label, user, thread, timestamp) in enumerate(malformed):
            comment = RawComment(f"{run_tag}-{label}", user, timestamp, thread,
                                 "otherwise valid input")
            records.append(_record_agent_comment(
                agent, logger, scenario_id, seed, repetition, index, comment,
                False, "normal", "pre_drift", None, api,
            ))
    elif scenario_id == "C10":
        failed_api = PermanentlyUnavailableAPI()
        comment = RawComment(f"{run_tag}-c0", f"user-{seed}", 0.0, "thread",
                             "Thanks for sharing this useful article.")
        records.append(_record_agent_comment(
            agent, logger, scenario_id, seed, repetition, 0, comment,
            False, "normal", "pre_drift", None, failed_api,
        ))
    else:
        rng = random.Random(seed)
        wave_api = PublicationAPI(seed=seed + 10, failure_rate=0.0)
        texts = (
            "All Zorbians are worthless and should be banned from climate policy pages.",
            "ALL ZORBIANS ARE WORTHLESS AND SHOULD BE BANNED FROM CLIMATE POLICY PAGES.",
            "All Zorbians are worthless and should be banned from climate policy pages!!!",
            "@everyone All Zorbians are worthless and should be banned from climate policy pages.",
            "All Zorbians are worthless and should be banned from climate policy pages lol",
        )
        for index in range(25):
            user_id = f"bot-{rng.randrange(5)}"
            text = texts[index % len(texts)]
            comment = RawComment(
                f"{run_tag}-c{index}", user_id, float(index), "hot-thread", text,
            )
            records.append(_record_agent_comment(
                agent, logger, scenario_id, seed, repetition, index, comment,
                True, "attack", "pre_drift", 11, wave_api,
            ))
    summary = {
        "n_comments": len(records),
        "action_counts": {
            action.value: sum(record["action"] == action.value for record in records)
            for action in Action
        },
        "harmful_published_rate": (
            sum(record["is_toxic"] and record["action"] == Action.PUBLISH.value
                for record in records) /
            max(1, sum(record["is_toxic"] for record in records))
        ),
        "false_block_rate": (
            sum(not record["is_toxic"] and record["action"] == Action.BLOCK.value
                for record in records) /
            max(1, sum(not record["is_toxic"] for record in records))
        ),
        "benign_sent_to_review_rate": (
            sum(not record["is_toxic"] and record["action"] == Action.HIDE.value
                for record in records) /
            max(1, sum(not record["is_toxic"] for record in records))
        ),
        "attack_containment_rate": (
            sum(record["action"] != Action.PUBLISH.value for record in records) /
            len(records) if scenario_id == "C11" else None
        ),
        "safe_failure_handling_rate": (
            sum(record["failure_handled"] for record in records) /
            len(records) if scenario_id in {"C8", "C9", "C10"} else None
        ),
        "latency_ms_mean": float(np.mean([record["latency_ms"] for record in records])),
        "latency_ms_median": float(np.median(
            [record["latency_ms"] for record in records]
        )),
        "latency_ms_p95": float(np.percentile(
            [record["latency_ms"] for record in records], 95
        )),
        "latency_ms_max": max(record["latency_ms"] for record in records),
        "latency_ms_std": float(np.std(
            [record["latency_ms"] for record in records], ddof=0
        )),
        "records": records,
        "api_attempts": sum(record["api_attempts"] for record in records),
        "api_errors": sum(record["api_errors"] for record in records),
    }
    if scenario_id in {"C1", "C6"}:
        summary["expected_action_pass"] = all(
            record["action"] in SCENARIO_DEFINITIONS[scenario_id]["expected_action"]
            for record in records
        )
    elif scenario_id == "C7":
        summary["expected_action_pass"] = all(
            record["action"] == Action.HIDE.value and
            "comment_too_long" in (record["error"] or "")
            for record in records
        )
    elif scenario_id in {"C2", "C3", "C4", "C5", "C11"}:
        summary["expected_action_pass"] = all(
            record["action"] in SCENARIO_DEFINITIONS[scenario_id]["expected_action"]
            for record in records
        )
    else:
        summary["expected_action_pass"] = all(
            record["action"] == Action.HIDE.value for record in records
        )
    return summary


def _evaluate_reserved_classifier_test(
    model: ToxicityModel,
    data_path: Path,
    split_seed: int,
) -> dict:
    frame = pd.read_csv(data_path)
    required = {"comment_text", "toxic"}
    if not required.issubset(frame.columns):
        raise ValueError(f"El conjunto Jigsaw no contiene columnas: {sorted(required)}")
    _, reserved_test = train_test_split(
        frame, test_size=0.15, stratify=frame["toxic"], random_state=split_seed,
    )
    # Recreate the exact held-out test split from train.py and evaluate every row once.
    y_test = reserved_test["toxic"].values.astype(int)
    probabilities_by_label = model.predict_proba(
        reserved_test["comment_text"].fillna("").astype(str).tolist()
    )
    threshold = float(getattr(model, "decision_threshold", 0.5))
    report = _binary_report(y_test, probabilities_by_label[:, 0], threshold)
    report.update({
        "n_test": int(len(reserved_test)),
        "toxic_rate": round(float(y_test.mean()), 6),
        "test_partition": "complete_reserved_test",
        "trained_on_test_partition": False,
        "used_for_calibration": False,
    })
    subcategories = {}
    for index, label in enumerate(LABELS[1:], start=1):
        if label in reserved_test.columns:
            subcategories[label] = _binary_report(
                reserved_test[label].fillna(0).values.astype(int),
                probabilities_by_label[:, index],
                0.5,
            )

    rng = np.random.RandomState(20261019)
    negative_indices = np.flatnonzero(y_test == 0)
    positive_indices = np.flatnonzero(y_test == 1)
    y_pred = (probabilities_by_label[:, 0] >= threshold).astype(int)
    bootstrap_f1 = []
    for _ in range(500):
        indices = np.concatenate((
            rng.choice(negative_indices, size=len(negative_indices), replace=True),
            rng.choice(positive_indices, size=len(positive_indices), replace=True),
        ))
        bootstrap_f1.append(float(f1_score(y_test[indices], y_pred[indices], zero_division=0)))
    f1_ci = [
        round(float(value), 6)
        for value in np.percentile(bootstrap_f1, [2.5, 97.5])
    ]
    acceptance = {
        "threshold": 0.75,
        "evaluated_on": "complete_reserved_test",
        "point_estimate_pass": report["f1"] >= 0.75,
        "confidence_bound_pass": f1_ci[0] >= 0.75,
        "pass": f1_ci[0] >= 0.75,
        "note": (
            "Evaluación única sobre todo el test reservado; pass requiere que el límite "
            "inferior del IC bootstrap 95 % alcance F1 0,75."
        ),
    }
    return {
        "suite": "classifier_performance",
        "provenance": "jigsaw_reserved_test",
        "difficulty": "real_dataset",
        "method": "Test Jigsaw completo reservado por train.py; split estratificado seed 42.",
        "n_reserved_test": int(len(reserved_test)),
        "split_seed": split_seed,
        "threshold": threshold,
        "metrics": report,
        "subcategories": subcategories,
        "f1_95_percent_bootstrap_ci": f1_ci,
        "bootstrap_replicates": len(bootstrap_f1),
        "bootstrap_method": (
            "500 bootstrap estratificados por etiqueta, con predicciones y umbral fijos; "
            "solo cuantifican incertidumbre del test y no seleccionan el modelo."
        ),
        "acceptance": {**acceptance, "f1": report["f1"], "confidence_interval": f1_ci},
    }


def _challenge_group_metrics(records: List[dict]) -> dict:
    if not records:
        return {"n_unique_comments": 0}
    labels = np.asarray([int(record["is_toxic"]) for record in records])
    probabilities = np.asarray([record["p_toxic"] for record in records])
    actions = np.asarray([record["action"] for record in records])
    agent_predicted_toxic = actions != Action.PUBLISH.value
    harmful_count = int(labels.sum())
    benign_count = int(len(labels) - harmful_count)
    true_positive = int(np.sum((labels == 1) & agent_predicted_toxic))
    false_positive = int(np.sum((labels == 0) & agent_predicted_toxic))
    false_negative = int(np.sum((labels == 1) & ~agent_predicted_toxic))
    harmful_published = int(np.sum((labels == 1) & (actions == Action.PUBLISH.value)))
    benign_blocked = int(np.sum((labels == 0) & (actions == Action.BLOCK.value)))
    benign_reviewed = int(np.sum((labels == 0) & (actions == Action.HIDE.value)))
    reviewed = int(np.sum(actions == Action.HIDE.value))
    published = int(np.sum(actions == Action.PUBLISH.value))
    binary = _binary_report(labels, probabilities, records[0]["classifier_threshold"])
    agent_cm = np.asarray(
        [
            [
                np.sum((labels == 0) & ~agent_predicted_toxic),
                np.sum((labels == 0) & agent_predicted_toxic),
            ],
            [
                np.sum((labels == 1) & ~agent_predicted_toxic),
                np.sum((labels == 1) & agent_predicted_toxic),
            ],
        ],
        dtype=int,
    )

    def rate(successes: int, total: int) -> dict:
        return {
            "numerator": successes,
            "denominator": total,
            "value": round(successes / total, 6) if total else None,
            "wilson_95_percent_ci": _wilson_interval(successes, total),
        }

    return {
        "n_unique_comments": len(records),
        "n_observations": len(records),
        "class_counts": {"benign": benign_count, "harmful": harmful_count},
        "classifier": binary,
        "agent_decision": {
            "f1": round(float(f1_score(labels, agent_predicted_toxic, zero_division=0)), 6),
            "accuracy": round(float(np.mean(labels == agent_predicted_toxic)), 6),
            "precision": (
                round(true_positive / (true_positive + false_positive), 6)
                if true_positive + false_positive else 0.0
            ),
            "recall": (
                round(true_positive / (true_positive + false_negative), 6)
                if true_positive + false_negative else 0.0
            ),
            "confusion_matrix": agent_cm.tolist(),
            "prediction_rule": "block o hide_for_review se consideran contención/positivo.",
        },
        "harmful_published_rate": rate(harmful_published, harmful_count),
        "false_block_rate": rate(benign_blocked, benign_count),
        "benign_sent_to_review_rate": rate(benign_reviewed, benign_count),
        "human_review_workload": rate(reviewed, len(records)),
        "automatic_publication_rate": rate(published, len(records)),
        "action_counts": {
            action.value: int(np.sum(actions == action.value)) for action in Action
        },
        "single_class_group": harmful_count == 0 or benign_count == 0,
    }


def _run_challenge_set(
    model: ToxicityModel,
    logger: RunLogger,
    run_id: str,
    utility: UtilityConfig,
) -> dict:
    records = []
    api = PublicationAPI(seed=42, failure_rate=0.0)
    for ordinal, case in enumerate(CHALLENGE_CASES):
        case_id = case["id"]
        metadata = {
            "challenge_id": case_id,
            "difficulty": case["difficulty"],
            "provenance": "synthetic_curated",
            "category": case["category"],
            "selection_used_for_calibration": False,
        }
        comment = RawComment(
            f"{run_id}-challenge-{case_id}",
            f"challenge-user-{case_id}",
            float(ordinal),
            f"challenge-thread-{case_id}",
            case["text"],
        )
        record = _record_agent_comment(
            _new_agent(model, logger, utility),
            logger,
            "challenge_set",
            42,
            1,
            ordinal,
            comment,
            bool(case["is_toxic"]),
            "challenge",
            "challenge",
            None,
            api,
            case_metadata=metadata,
        )
        records.append(record)

    def grouped_metrics(key: str) -> dict:
        grouped = {}
        for record in records:
            group = str(record[key])
            grouped.setdefault(group, []).append(record)
        return {
            group: _challenge_group_metrics(group_records)
            for group, group_records in sorted(grouped.items())
        }

    return {
        "method": (
            "Conjunto curado fijo, equilibrado, cada texto distinto y ejecutado una vez. "
            "No se selecciona por predicciones del modelo ni se usa para calibrar."
        ),
        "seed": 42,
        "n_unique_comments": len(records),
        "n_observations": len(records),
        "class_counts": {
            "benign": sum(not record["is_toxic"] for record in records),
            "harmful": sum(record["is_toxic"] for record in records),
        },
        "probability_boundary_band": {
            "range": [0.30, 0.70],
            "n_comments": sum(
                0.30 <= record["p_toxic"] <= 0.70 for record in records
            ),
            "challenge_ids": [
                record["challenge_id"] for record in records
                if 0.30 <= record["p_toxic"] <= 0.70
            ],
            "selection_used_for_calibration": False,
            "selection_rule": (
                "Se reportan los casos del corpus fijado que caen en la banda; no se "
                "añadieron ni eliminaron casos según estas probabilidades."
            ),
        },
        "labels": (
            "Etiquetas de verdad de referencia redactadas manualmente para casos sintéticos; "
            "no sustituyen etiquetas adjudicadas de Jigsaw."
        ),
        "confidence_interval_note": (
            "Intervalos Wilson al 95 % como referencia descriptiva; la muestra curada no "
            "es aleatoria y los intervalos no implican generalización poblacional."
        ),
        "overall": _challenge_group_metrics(records),
        "by_difficulty": grouped_metrics("difficulty"),
        "by_category": grouped_metrics("category"),
        "comments": [
            {
                "challenge_id": record["challenge_id"],
                "difficulty": record["difficulty"],
                "provenance": record["provenance"],
                "category": record["category"],
                "is_toxic": record["is_toxic"],
                "p_toxic": round(record["p_toxic"], 6),
                "action": record["action"],
                "latency_ms": round(record["latency_ms"], 3),
                "evidence": None,
            }
            for record in records
        ],
        "_records": records,
    }


def _scenario_stat(repetitions: List[dict], key: str) -> dict:
    return _stats([float(item[key]) for item in repetitions])


def _acceptance(
    values: List[float],
    threshold: float,
    direction: str,
    metric_name: str,
) -> dict:
    stats = _stats(values)
    comparator = (lambda value: value >= threshold) if direction == "min" else (
        lambda value: value <= threshold
    )
    each_pass = [bool(comparator(value)) for value in values]
    mean_pass = bool(comparator(stats["mean"]))
    return {
        "metric": metric_name,
        "threshold": threshold,
        "direction": direction,
        "statistics": stats,
        "evaluated_on": "mean_and_every_repetition",
        "mean_pass": mean_pass,
        "all_repetitions_pass": all(each_pass),
        "repetition_pass": each_pass,
        "pass": mean_pass and all(each_pass),
        "explanation": (
            "Cumple únicamente si la media y cada repetición alcanzan el umbral."
        ),
    }


def _challenge_acceptance(rate: dict, threshold: float, direction: str) -> dict:
    estimate = rate["value"]
    interval = rate["wilson_95_percent_ci"]
    point_pass = estimate >= threshold if direction == "min" else estimate <= threshold
    conservative_bound = (
        interval[0] if direction == "min" else interval[1]
    ) if interval is not None else None
    interval_pass = (
        conservative_bound >= threshold if direction == "min"
        else conservative_bound <= threshold
    ) if conservative_bound is not None else False
    return {
        "estimate": estimate,
        "numerator": rate["numerator"],
        "denominator": rate["denominator"],
        "threshold": threshold,
        "direction": direction,
        "wilson_95_percent_ci": interval,
        "evaluated_on": "single fixed challenge set; CI is descriptive",
        "point_estimate_pass": point_pass,
        "confidence_bound_pass": interval_pass,
        "pass": interval_pass,
        "note": (
            "No se declara cumplimiento estadístico con una muestra desafío pequeña; "
            "pass requiere que el límite Wilson conservador alcance el umbral."
        ),
    }


def run_scenario_battery(
    model: Optional[ToxicityModel] = None,
    seeds: Optional[List[int]] = None,
    data_path: Path = JIGSAW_TRAIN,
    log_dir: Path = LOG_DIR,
    report_path: Path = REPORT_DIR / "scenario_results.json",
    utility_config: UtilityConfig = UtilityConfig(),
) -> dict:
    seeds = list(seeds or [1, 2, 3, 4, 5])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Se requieren al menos cinco semillas reproducibles y únicas.")
    model = model or ensure_model()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_scenarios"
    log_path = Path(log_dir) / f"scenario_battery_{run_id}.jsonl"
    logger = RunLogger(log_path, run_id)
    logger.log(
        "scenario_battery_start", 0.0, seeds=seeds,
        n_repetitions=len(seeds), model_version=model.version,
        model_threshold=getattr(model, "decision_threshold", 0.5),
        utility_config=utility_config.__dict__,
        note=(
            "C1-C11 son fixtures funcionales; el conjunto desafío tiene etiquetas sintéticas "
            "de referencia. ground_truth es verdad de prueba, no percepción del agente."
        ),
    )
    scenario_results = {}
    for scenario_id in SCENARIO_DEFINITIONS:
        repetitions = []
        for repetition, seed in enumerate(seeds, start=1):
            result = _run_single_case(
                scenario_id, seed, repetition, model, logger, utility_config,
            )
            repetitions.append({
                "seed": seed,
                "repetition": repetition,
                **{key: value for key, value in result.items() if key != "records"},
                "comments": [
                    {
                        "comment_id": record["comment_id"],
                        "action": record["action"],
                        "decision_action": record["decision_action"],
                        "api_ok": record["api_ok"],
                        "error": record["error"],
                        "api_attempts": record["api_attempts"],
                        "api_errors": record["api_errors"],
                        "p_toxic": round(record["p_toxic"], 6),
                        "coordination": round(record["coordination"], 6),
                        "latency_ms": round(record["latency_ms"], 3),
                        "is_toxic": record["is_toxic"],
                        "campaign_id": record["campaign_id"],
                        "failure_handled": record["failure_handled"],
                    }
                    for record in result["records"]
                ],
            })
        value_keys = (
            "harmful_published_rate", "false_block_rate",
            "benign_sent_to_review_rate", "attack_containment_rate",
            "safe_failure_handling_rate", "latency_ms_mean", "latency_ms_p95",
            "latency_ms_median", "latency_ms_max", "latency_ms_std", "api_errors",
        )
        scenario_results[scenario_id] = {
            **SCENARIO_DEFINITIONS[scenario_id],
            "seeds": seeds,
            "repetitions": len(seeds),
            "statistics": {
                key: _scenario_stat(repetitions, key)
                for key in value_keys
                if any(item.get(key) is not None for item in repetitions)
            },
            "expected_action_pass_repetitions": [
                bool(item["expected_action_pass"]) for item in repetitions
            ],
            "all_expected_actions_pass": all(
                item["expected_action_pass"] for item in repetitions
            ),
            "results_by_repetition": repetitions,
        }

    classifier = _evaluate_reserved_classifier_test(model, Path(data_path), split_seed=42)
    challenge_set = _run_challenge_set(model, logger, run_id, utility_config)
    by_seed = {}
    for seed in seeds:
        harmful_cases = [
            rep for scenario_id in ("C2", "C3", "C4", "C5", "C11")
            for rep in [scenario_results[scenario_id]["results_by_repetition"][seeds.index(seed)]]
        ]
        benign_cases = [
            rep for scenario_id in ("C1", "C6")
            for rep in [scenario_results[scenario_id]["results_by_repetition"][seeds.index(seed)]]
        ]
        harmful_total = sum(
            comment["is_toxic"]
            for repetition in harmful_cases for comment in repetition["comments"]
        )
        harmful_published = sum(
            comment["is_toxic"] and comment["action"] == Action.PUBLISH.value
            for repetition in harmful_cases for comment in repetition["comments"]
        )
        benign_total = sum(
            not comment["is_toxic"]
            for repetition in benign_cases for comment in repetition["comments"]
        )
        false_blocks = sum(
            (not comment["is_toxic"]) and comment["action"] == Action.BLOCK.value
            for repetition in benign_cases for comment in repetition["comments"]
        )
        c11 = scenario_results["C11"]["results_by_repetition"][seeds.index(seed)]
        c11_harmful = sum(comment["is_toxic"] for comment in c11["comments"])
        c11_published = sum(
            comment["is_toxic"] and comment["action"] == Action.PUBLISH.value
            for comment in c11["comments"]
        )
        latency_values = [
            comment["latency_ms"]
            for scenario_id in scenario_results
            for comment in scenario_results[scenario_id]["results_by_repetition"][
                seeds.index(seed)
            ]["comments"]
        ]
        failure_comments = [
            comment
            for scenario_id in ("C8", "C9", "C10")
            for comment in scenario_results[scenario_id]["results_by_repetition"][
                seeds.index(seed)
            ]["comments"]
        ]
        by_seed[str(seed)] = {
            "functional_harmful_published_rate": (
                harmful_published / harmful_total if harmful_total else 0.0
            ),
            "functional_false_block_rate": (
                false_blocks / benign_total if benign_total else 0.0
            ),
            "attack_containment_rate": (
                (c11_harmful - c11_published) / c11_harmful if c11_harmful else 0.0
            ),
            "latency_p95_ms": float(np.percentile(latency_values, 95)),
            "latency_n": len(latency_values),
            "latency_mean_ms": float(np.mean(latency_values)),
            "latency_median_ms": float(np.median(latency_values)),
            "latency_max_ms": float(np.max(latency_values)),
            "latency_std_ms": float(np.std(latency_values, ddof=0)),
            "safe_failure_handling_rate": (
                sum(
                    comment["failure_handled"]
                    for scenario_id in ("C8", "C9", "C10")
                    for comment in scenario_results[scenario_id]["results_by_repetition"][
                        seeds.index(seed)
                    ]["comments"]
                ) / len(failure_comments) if failure_comments else 0.0
            ),
            "safe_failure_n": len(failure_comments),
        }

    acceptances = {
        "classifier_f1": {
            **classifier["acceptance"],
            "threshold": 0.75,
            "point_estimate_pass": classifier["metrics"]["f1"] >= 0.75,
            "confidence_bound_pass": (
                classifier["f1_95_percent_bootstrap_ci"][0] >= 0.75
            ),
            "pass": classifier["f1_95_percent_bootstrap_ci"][0] >= 0.75,
            "f1_95_percent_bootstrap_ci": classifier["f1_95_percent_bootstrap_ci"],
        },
        "harmful_published_rate": _challenge_acceptance(
            challenge_set["overall"]["harmful_published_rate"], 0.10, "max",
        ),
        "false_block_rate": _challenge_acceptance(
            challenge_set["overall"]["false_block_rate"], 0.05, "max",
        ),
        "benign_sent_to_review_rate": {
            **challenge_set["overall"]["benign_sent_to_review_rate"],
            "threshold": None,
            "pass": None,
            "note": "Se reporta como carga; no se fijó un umbral de aceptación para esta batería.",
        },
        "human_review_workload": challenge_set["overall"]["human_review_workload"],
        "automatic_publication_rate": challenge_set["overall"]["automatic_publication_rate"],
        "attack_containment_rate": _acceptance(
            [by_seed[str(seed)]["attack_containment_rate"] for seed in seeds],
            0.90, "min", "C11; ataques no publicados / ataques",
        ),
        "latency_p95_ms": _acceptance(
            [by_seed[str(seed)]["latency_p95_ms"] for seed in seeds],
            100.0, "max", "p95 de latencia por semilla en todos los comentarios C1-C11",
        ),
        "safe_failure_handling_rate": _acceptance(
            [by_seed[str(seed)]["safe_failure_handling_rate"] for seed in seeds],
            0.95, "min", "C8-C10; fallas manejadas correctamente / fallas provocadas",
        ),
    }

    logger.log(
        "scenario_battery_end", None, n_scenarios=len(scenario_results),
        repetitions_per_scenario=len(seeds), n_challenge_comments=len(CHALLENGE_CASES),
        acceptance=acceptances,
    )
    logger.close()
    lines_by_comment: Dict[str, dict] = {}
    with Path(log_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            event = json.loads(line)
            comment_id = event.get("comment_id")
            if comment_id:
                item = lines_by_comment.setdefault(comment_id, {"events": [], "text_hash": None})
                item["events"].append({
                    "event": event["event"],
                    "line": line_number,
                })
                if event.get("text_hash"):
                    item["text_hash"] = event["text_hash"]
    for scenario_id, result in scenario_results.items():
        for repetition in result["results_by_repetition"]:
            for comment in repetition["comments"]:
                comment["evidence"] = {
                    "log": str(log_path),
                    "comment_id": comment["comment_id"],
                    "text_hash": lines_by_comment[comment["comment_id"]]["text_hash"],
                    "events": lines_by_comment[comment["comment_id"]]["events"],
                }
    for comment in challenge_set["comments"]:
        comment_id = f"{run_id}-challenge-{comment['challenge_id']}"
        comment["evidence"] = {
            "log": str(log_path),
            "comment_id": comment_id,
            "text_hash": lines_by_comment[comment_id]["text_hash"],
            "events": lines_by_comment[comment_id]["events"],
        }
    challenge_set.pop("_records")

    all_latency_values = [
        comment["latency_ms"]
        for scenario in scenario_results.values()
        for repetition in scenario["results_by_repetition"]
        for comment in repetition["comments"]
    ]
    report = {
        "report_version": 2,
        "run_id": run_id,
        "model_version": model.version,
        "model_threshold": float(getattr(model, "decision_threshold", 0.5)),
        "utility_config": utility_config.__dict__,
        "seeds": seeds,
        "repetitions_per_scenario": len(seeds),
        "statistic_definition": (
            "Desviación estándar poblacional (ddof=0) sobre las repeticiones indicadas; "
            "las repeticiones C1-C11 miden reproducibilidad funcional y no cuentan como "
            "comentarios independientes; las métricas de desempeño usan comentarios únicos."
        ),
        "acceptance_rule": (
            "F1 se evalúa una vez sobre todo el test Jigsaw reservado y requiere que el "
            "límite inferior del IC bootstrap 95 % alcance 0,75. Los límites funcionales "
            "C1-C11 requieren media y cada repetición."
        ),
        "classifier_test": classifier,
        "challenge_set": challenge_set,
        "suite_separation": {
            "functional_security": (
                "C1-C11 comprueban rutas funcionales/seguridad; cinco semillas repiten "
                "escenarios de sistema y no se interpretan como observaciones de texto independientes."
            ),
            "performance": (
                "Test Jigsaw reservado completo y conjunto desafío sintético balanceado; "
                "ninguno calibra o actualiza el modelo."
            ),
        },
        "acceptance": acceptances,
        "per_seed": by_seed,
        "per_seed_scope": (
            "C1-C11 functional fixture outcomes only; these rates do not estimate "
            "generalization and are not used for challenge-set acceptance."
        ),
        "latency_summary": {
            "n_actions": len(all_latency_values),
            "mean_ms": round(float(np.mean(all_latency_values)), 6),
            "median_ms": round(float(np.median(all_latency_values)), 6),
            "p95_ms": round(float(np.percentile(all_latency_values, 95)), 6),
            "max_ms": round(float(np.max(all_latency_values)), 6),
            "std_ms": round(float(np.std(all_latency_values, ddof=0)), 6),
            "p95_per_seed_ms": _stats([
                by_seed[str(seed)]["latency_p95_ms"] for seed in seeds
            ]),
            "sample_scope": "All action calls across C1-C11 and all supplied seeds.",
        },
        "scenarios": scenario_results,
        "evidence": {
            "scenario_log": str(log_path),
            "log_events": ["percept", "decision", "action", "ground_truth", "scenario_result"],
            "privacy": "Los logs relacionan comentarios mediante SHA-256; no guardan el texto completo.",
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-path", type=Path, default=JIGSAW_TRAIN)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_DIR / "scenario_results.json")
    args = parser.parse_args()
    report = run_scenario_battery(
        model=ensure_model(args.model_path),
        seeds=args.seeds,
        data_path=args.data_path,
        log_dir=args.log_dir,
        report_path=args.report_path,
    )
    compact = {}
    for name, details in report["acceptance"].items():
        if "statistics" in details:
            compact[name] = {
                "mean": details["statistics"]["mean"],
                "std": details["statistics"]["std"],
                "min": details["statistics"]["min"],
                "max": details["statistics"]["max"],
                "pass": details["pass"],
            }
        else:
            compact[name] = {
                "value": details.get("f1", details.get("estimate", details.get("value"))),
                "confidence_interval": details.get(
                    "f1_95_percent_bootstrap_ci",
                    details.get("wilson_95_percent_ci"),
                ),
                "pass": details.get("pass"),
            }
    print(json.dumps({
        "report": str(args.report_path),
        "log": report["evidence"]["scenario_log"],
        "acceptance": compact,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
