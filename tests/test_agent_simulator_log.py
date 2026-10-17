import dataclasses
import json
from pathlib import Path

from safefeed.agent import SafeFeedAgent
from safefeed.config import LABELS, AgentConfig
from safefeed.logger import RunLogger
from safefeed.runner import run_simulation
from safefeed.schemas import Action, ModeratorFeedback, Percept, RawComment
from safefeed.simulator import ModeratorSimulator, PlatformSimulator, PublicationAPI


class AlwaysOkAPI:
    def __init__(self):
        self.calls = []

    def execute(self, action, comment_id, sim_time):
        self.calls.append((action, comment_id))
        from safefeed.simulator import ApiResponse
        return ApiResponse(True, "ok")


# ------------------------------------------------------------ ciclo del agente
def test_percept_has_no_ground_truth_fields():
    names = {f.name for f in dataclasses.fields(Percept)}
    assert not names & {"is_toxic", "labels", "truth", "kind", "campaign_id"}


def test_step_runs_perceive_decide_act_in_order_and_logs(tmp_path, base_model):
    logger = RunLogger(tmp_path / "t.jsonl", "t")
    agent = SafeFeedAgent(base_model, logger=logger)
    api = AlwaysOkAPI()
    raw = RawComment("c1", "u1", 5.0, "t1", "You are a total idiot, nobody cares.")
    percept, decision, result = agent.step(raw, api)
    logger.close()
    events = [json.loads(l)["event"] for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert events == ["percept", "decision", "action"]
    assert decision.action in set(Action) and result.ok
    assert api.calls == [(decision.action, "c1")]          # la acción llegó a la API
    assert decision.action != Action.PUBLISH               # insulto evidente no se publica


def test_benign_comment_is_published(base_model):
    agent = SafeFeedAgent(base_model)
    _, decision, _ = agent.step(
        RawComment("c2", "u2", 1.0, "t1", "Thanks for fixing the references in the jazz history section."),
        AlwaysOkAPI())
    assert decision.action == Action.PUBLISH


def test_every_log_line_has_timestamp_and_event(tmp_path, base_model):
    out = run_simulation(200, seed=5, model=base_model, log_dir=tmp_path, api_failure_rate=0.0)
    lines = [json.loads(l) for l in Path(out["log_path"]).read_text().splitlines()]
    assert lines[0]["event"] == "run_start" and lines[-1]["event"] == "run_end"
    assert all("ts" in l and "event" in l and "run_id" in l for l in lines)
    # por comentario: percepción antes que decisión antes que acción
    order = {}
    for i, l in enumerate(lines):
        if l["event"] in ("percept", "decision", "action"):
            order.setdefault(l["comment_id"], {})[l["event"]] = i
    assert all(o["percept"] < o["decision"] < o["action"] for o in order.values())
    assert len(order) == 200


# ------------------------------------------------------------ simulador
def test_simulator_is_reproducible_per_seed_and_varies_across_seeds():
    a = [(r.text, r.user_id) for r, _ in PlatformSimulator(300, seed=1)]
    b = [(r.text, r.user_id) for r, _ in PlatformSimulator(300, seed=1)]
    c = [(r.text, r.user_id) for r, _ in PlatformSimulator(300, seed=2)]
    assert a == b and a != c


def test_simulator_has_attack_waves_and_drift():
    items = list(PlatformSimulator(2000, seed=4))
    kinds = {t.kind for _, t in items}
    phases = {t.phase for _, t in items}
    assert kinds == {"normal", "attack"} and phases == {"pre_drift", "post_drift"}
    attacks = [r for r, t in items if t.kind == "attack"]
    assert len({r.user_id for r in attacks}) > 3            # varias cuentas coordinadas


def test_api_failures_are_stochastic():
    api = PublicationAPI(seed=1, failure_rate=0.5)
    oks = [api.execute(Action.PUBLISH, f"c{i}", 0.0).ok for i in range(200)]
    assert 0 < sum(oks) < 200


def test_moderator_only_reviews_hidden_and_some_blocks():
    mods = ModeratorSimulator(seed=1, appeal_rate=0.0)
    from safefeed.simulator import GroundTruth
    truth = GroundTruth({k: 0 for k in LABELS} | {"toxic": 1}, True, "normal", "pre_drift")
    mods.consider("c1", Action.HIDE, truth, 0.0)
    mods.consider("c2", Action.BLOCK, truth, 0.0)       # appeal_rate = 0 -> no apelación
    mods.consider("c3", Action.PUBLISH, truth, 0.0)
    assert [f.comment_id for f in mods.due(1e9)] == ["c1"]


# ------------------------------------------------------------ aprendizaje
def test_feedback_triggers_retraining_and_raises_probability(base_model):
    import copy
    model = copy.deepcopy(base_model)
    agent = SafeFeedAgent(model, AgentConfig(retrain_every=40, feedback_weight=3.0))
    novel = "Seriously numbskull, learn to read before you touch coffee farming again."
    p_before = model.predict_proba([novel])[0][0]
    v_before = model.version
    for i in range(40):
        cid = f"f{i}"
        agent._pending[cid] = novel if i % 2 == 0 else novel.replace("Seriously", "Honestly")
        labels = {k: 0 for k in LABELS} | {"toxic": 1, "insult": 1}
        agent.receive_feedback(ModeratorFeedback(cid, "confirmed", labels, float(i), "review_queue"))
    assert model.version == v_before + 1 and agent.n_retrains == 1
    assert model.predict_proba([novel])[0][0] > p_before


def test_learning_disabled_never_retrains(base_model):
    import copy
    model = copy.deepcopy(base_model)
    agent = SafeFeedAgent(model, AgentConfig(retrain_every=5, learning_enabled=False))
    for i in range(20):
        agent._pending[f"c{i}"] = "some text"
        agent.receive_feedback(ModeratorFeedback(f"c{i}", "confirmed", {k: 0 for k in LABELS}, 0.0, "appeal"))
    assert model.version == base_model.version and agent.n_retrains == 0


# ------------------------------------------------------------ reproducibilidad de punta a punta
def test_same_seed_gives_identical_decisions(base_model):
    cfg = AgentConfig(retrain_every=30)
    a = run_simulation(400, seed=9, model=base_model, agent_config=cfg, log_dir=None)
    b = run_simulation(400, seed=9, model=base_model, agent_config=cfg, log_dir=None)
    assert [r["action"] for r in a["records"]] == [r["action"] for r in b["records"]]
    assert [round(r["p_toxic"], 9) for r in a["records"]] == [round(r["p_toxic"], 9) for r in b["records"]]


def test_run_does_not_mutate_the_shared_model(base_model):
    v = base_model.version
    run_simulation(300, seed=2, model=base_model, agent_config=AgentConfig(retrain_every=10), log_dir=None)
    assert base_model.version == v


def test_attack_waves_are_contained(base_model):
    out = run_simulation(1500, seed=6, model=base_model, log_dir=None)
    att = out["summary"]["attacks"]
    assert att["contained_rate"] >= 0.9
