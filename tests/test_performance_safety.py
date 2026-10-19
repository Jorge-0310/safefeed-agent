import copy
import json

import numpy as np
import pytest

from safefeed.agent import SafeFeedAgent
from safefeed.config import CLASSIFIER_THRESHOLDS, LABELS, AgentConfig, UtilityConfig
from safefeed.logger import RunLogger
from safefeed.metrics import calibrate_utility_policy, compute_run_metrics
from safefeed.runner import evaluate_failure_handling
from safefeed.schemas import Action, ModeratorFeedback, Percept, RawComment
from safefeed.simulator import ApiResponse
from safefeed.train import calibrate_threshold


class AlwaysOkAPI:
    def execute(self, action, comment_id, sim_time):
        return ApiResponse(True, "ok")


class InvalidAPI:
    def execute(self, action, comment_id, sim_time):
        return {"ok": True, "status": "not-an-api-response"}


class UnavailableAPI:
    def execute(self, action, comment_id, sim_time):
        raise ConnectionError("offline")


class FixedModel:
    version = 1
    decision_threshold = 0.5

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, texts):
        return np.tile(self.probabilities, (len(texts), 1))


@pytest.mark.parametrize(
    "comment",
    [
        RawComment("empty", "u", 1.0, "t", ""),
        RawComment("spaces", "u", 1.0, "t", "   \t"),
        RawComment("absent-text", "u", 1.0, "t", None),
        RawComment("absent-user", "", 1.0, "t", "a valid message"),
        RawComment("absent-thread", "u", 1.0, "", "a valid message"),
        RawComment("bad-time", "u", float("nan"), "t", "a valid message"),
        RawComment("huge-time", "u", 10**1000, "t", "a valid message"),
        RawComment("bad-format", "u", 1.0, "t", 17),
    ],
)
def test_invalid_comment_fails_closed_to_review(comment, base_model):
    agent = SafeFeedAgent(base_model)
    percept, decision, result = agent.step(comment, AlwaysOkAPI())
    assert decision.action == Action.HIDE
    assert result.ok and result.action == Action.HIDE
    assert decision.error


def test_absent_comment_and_coordination_sensor_data_fail_closed(base_model):
    agent = SafeFeedAgent(base_model)
    _, decision, result = agent.step(None, AlwaysOkAPI())
    assert decision.action == Action.HIDE and result.ok

    agent = SafeFeedAgent(base_model)
    agent.memory.features = lambda *args: {}
    _, decision, result = agent.step(
        RawComment("sensor", "u", 1.0, "t", "a valid comment"), AlwaysOkAPI()
    )
    assert decision.action == Action.HIDE and result.ok
    assert "coordination_sensor_missing_data" in decision.error


def test_failure_injection_matrix_handles_all_declared_failure_types(base_model):
    report = evaluate_failure_handling(base_model)
    assert report["failures_injected"] == 11
    assert report["safe_failure_handling_rate"] == 1.0
    assert all(case["handled_safely"] for case in report["cases"])


def test_invalid_input_errors_are_logged_without_raw_text(base_model, tmp_path):
    logger = RunLogger(tmp_path / "invalid.jsonl", "invalid")
    agent = SafeFeedAgent(base_model, logger=logger)
    agent.step(RawComment("blank", "u", 1.0, "t", "  "), AlwaysOkAPI())
    logger.close()
    events = [json.loads(line) for line in (tmp_path / "invalid.jsonl").read_text().splitlines()]
    assert events[0]["error"] == "text_whitespace"
    assert events[1]["safety_rule"] == "invalid_input"
    assert events[1]["error"] == "text_whitespace"
    assert events[0]["text_hash"] and "text" not in events[0]


@pytest.mark.parametrize("api", [InvalidAPI(), UnavailableAPI()])
def test_invalid_or_unavailable_api_never_falls_back_to_publish(api, base_model):
    agent = SafeFeedAgent(base_model)
    _, decision, result = agent.step(
        RawComment("api-failure", "u", 1.0, "t", "Thanks for the useful article."),
        api,
    )
    assert result.action == Action.HIDE
    assert not result.ok
    assert result.error
    assert decision.action != Action.BLOCK


def test_severe_safety_rules_override_expected_utility():
    threat_model = FixedModel([0.4, 0.0, 0.0, 0.91, 0.0, 0.0])
    agent = SafeFeedAgent(threat_model)
    percept = Percept("threat", "u", 1.0, "t", "text", "text", [], {
        "score": 0.0, "n_similar_recent": 0, "distinct_users_similar": 0,
        "same_user_last_60s": 0, "window_size": 0,
    })
    decision = agent.decide(percept)
    assert decision.action == Action.BLOCK
    assert decision.safety_rule == "explicit_threat"

    hate_model = FixedModel([0.4, 0.0, 0.0, 0.0, 0.0, 0.95])
    hate_agent = SafeFeedAgent(hate_model)
    decision = hate_agent.decide(percept)
    assert decision.action == Action.HIDE
    assert decision.safety_rule == "identity_hate"

    severe_model = FixedModel([0.4, 0.96, 0.0, 0.0, 0.0, 0.0])
    severe_agent = SafeFeedAgent(severe_model)
    decision = severe_agent.decide(percept)
    assert decision.action == Action.BLOCK
    assert decision.safety_rule == "severe_toxicity"


def test_high_confidence_coordinated_attack_is_blocked():
    agent = SafeFeedAgent(FixedModel([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]))
    percept = Percept("attack", "bot", 10.0, "thread", "repeated attack", "repeated attack", [], {
        "score": 0.9, "n_similar_recent": 5, "distinct_users_similar": 5,
        "same_user_last_60s": 0, "window_size": 5,
    })
    decision = agent.decide(percept)
    assert decision.action == Action.BLOCK
    assert decision.safety_rule == "coordinated_attack"


def test_evasive_spelling_is_not_allowed_to_bypass_character_model(base_model):
    agent = SafeFeedAgent(base_model)
    text = "You are a total 1d10t, nobody cares about your climate policy edits."
    _, decision, _ = agent.step(
        RawComment("evasive", "u", 1.0, "t", text), AlwaysOkAPI()
    )
    assert decision.p_toxic > 0.0
    assert decision.action != Action.PUBLISH

def test_false_negative_post_publication_audit_is_learned_and_changes_model(
    base_model, tmp_path
):
    model = copy.deepcopy(base_model)
    logger = RunLogger(tmp_path / "feedback.jsonl", "feedback")
    agent = SafeFeedAgent(
        model,
        AgentConfig(retrain_every=40, feedback_weight=3.0),
        UtilityConfig(publish_harmful=-1.5),
        logger=logger,
    )
    text = "dingbat, go away"
    before = float(model.predict_proba([text])[0, 0])
    percept = Percept("initial", "u", 1.0, "t", text, text, [], {
        "score": 0.0, "n_similar_recent": 0, "distinct_users_similar": 0,
        "same_user_last_60s": 0, "window_size": 0,
    })
    before_action = agent.decide(percept).action
    assert before_action == Action.PUBLISH
    labels = {name: 0 for name in LABELS}
    labels["toxic"] = 1
    labels["insult"] = 1

    for i in range(40):
        comment_id = f"published-fn-{i}"
        agent._pending[comment_id] = (text, Action.PUBLISH)
        agent.receive_feedback(
            ModeratorFeedback(
                comment_id, "false_negative", labels, float(i), "post_publication_audit"
            )
        )
    logger.close()

    events = [json.loads(line) for line in (tmp_path / "feedback.jsonl").read_text().splitlines()]
    fn_events = [event for event in events if event.get("classification") == "false_negative"]
    retrains = [event for event in events if event["event"] == "retrain"]
    after = float(model.predict_proba([text])[0, 0])
    assert fn_events and fn_events[0]["ground_truth"]["source"] == "moderator_feedback"
    assert agent.n_retrains == 1 and retrains
    assert len(model._texts) > model._base_size
    assert after > before

    next_percept = Percept("later", "u", 100.0, "t", text, text, [], {
        "score": 0.0, "n_similar_recent": 0, "distinct_users_similar": 0,
        "same_user_last_60s": 0, "window_size": 0,
    })
    assert agent.decide(next_percept).action != Action.PUBLISH


def test_threshold_calibration_uses_validation_candidates_only():
    y = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.01, 0.2, 0.8, 0.9])
    calibration = calibrate_threshold(y, scores)
    assert calibration["selected_threshold"] in CLASSIFIER_THRESHOLDS
    assert calibration["selection_constraints"]["selection_split"] == "validation"
    assert len(calibration["threshold_sweep"]) == len(CLASSIFIER_THRESHOLDS)


def test_classifier_and_agent_metrics_are_reported_separately():
    summary = compute_run_metrics([
        {"is_toxic": 1, "p_toxic": 0.8, "action": "publish", "latency_ms": 1,
         "kind": "normal", "phase": "pre_drift", "classifier_threshold": 0.5},
        {"is_toxic": 0, "p_toxic": 0.1, "action": "block", "latency_ms": 3,
         "kind": "normal", "phase": "post_drift", "classifier_threshold": 0.5},
        {"is_toxic": 0, "p_toxic": 0.9, "action": "hide_for_review", "latency_ms": 2,
         "kind": "normal", "phase": "post_drift", "classifier_threshold": 0.5},
    ])
    assert summary["classifier"]["confusion_matrix"] == [[1, 1], [0, 1]]
    assert summary["agent"]["false_negative"] == 1
    assert summary["agent"]["false_blocks"] == 1
    assert summary["agent"]["benign_sent_to_review"] == 1
    assert summary["by_phase"]["pre_drift"]["n"] == 1
    assert summary["classifier"]["by_phase"]["post_drift"]["n"] == 2
    assert summary["by_model_version"]["0"]["n"] == 3
    assert summary["latency_ms_median"] == 2
    assert "auc_pr" in summary["classifier"]
    assert "attacks" in summary["simulator"]


def test_utility_calibration_selects_safest_feasible_region():
    records = [
        {"is_toxic": True, "p_toxic": 0.05, "coordination": 0.0,
         "subcategory_probs": {}, "kind": "normal", "phase": "pre_drift",
         "action": "publish", "latency_ms": 1.0}
        for _ in range(100)
    ] + [
        {"is_toxic": False, "p_toxic": 0.0, "coordination": 0.0,
         "subcategory_probs": {}, "kind": "normal", "phase": "pre_drift",
         "action": "publish", "latency_ms": 1.0}
        for _ in range(100)
    ]
    result = calibrate_utility_policy(records)
    assert result["selection_split"] == "validation_simulation"
    assert result["selected_publish_harmful"] != -1.5
    selected = next(
        candidate for candidate in result["candidates"]
        if candidate["publish_harmful"] == result["selected_publish_harmful"]
    )
    assert selected["meets_safety_targets"]
    assert result["selected_metrics"]["agent"]["harmful_published_rate"] <= 0.10
    assert result["selected_metrics"]["agent"]["false_block_rate"] <= 0.05
