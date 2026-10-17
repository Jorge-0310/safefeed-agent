from safefeed.config import UtilityConfig
from safefeed.coordination import ShortTermMemory
from safefeed.schemas import Action
from safefeed.text import clean_text, tokenize
from safefeed.utility import (choose_action, effective_probability, expected_utilities,
                              severity_multiplier)

U = UtilityConfig()


def _act(p, coord=0.0, sev=1.0, surge=False):
    p_eff = effective_probability(p, coord, U)
    return choose_action(expected_utilities(p_eff, sev, surge, U))


# ------------------------------------------------------------------ texto
def test_clean_text_is_idempotent_and_normalizes():
    raw = "  You   are SOOOO bad!!! http://spam.example/x  "
    once = clean_text(raw)
    assert once == clean_text(once)
    assert "http" not in once and "ooo" not in once and once == once.lower()


def test_tokenize_basic():
    assert tokenize("Hello, World! it's") == ["hello", "world", "it's"]


# ------------------------------------------------------------------ utilidad
def test_action_regions_follow_probability():
    assert _act(0.01) == Action.PUBLISH
    assert _act(0.30) == Action.HIDE
    assert _act(0.95) == Action.BLOCK


def test_action_is_monotone_in_probability():
    order = {Action.PUBLISH: 0, Action.HIDE: 1, Action.BLOCK: 2}
    ranks = [order[_act(p / 100)] for p in range(0, 101)]
    assert ranks == sorted(ranks)   # nunca "baja" de severidad al subir p


def test_surge_never_makes_agent_more_permissive():
    order = {Action.PUBLISH: 0, Action.HIDE: 1, Action.BLOCK: 2}
    for p in [0.02, 0.08, 0.12, 0.3, 0.6, 0.8]:
        assert order[_act(p, surge=True)] >= order[_act(p, surge=False)]


def test_surge_changes_a_borderline_case():
    # p = 0.09: sin ola se publica; en una ola, dejar pasar daño duele más y se oculta.
    assert _act(0.09, surge=False) == Action.PUBLISH
    assert _act(0.09, surge=True) == Action.HIDE


def test_severity_multiplier_range():
    assert severity_multiplier({"threat": 0.0}, U) == 1.0
    assert abs(severity_multiplier({"threat": 1.0}, U) - (1.0 + U.severity_gain)) < 1e-9


def test_coordination_needs_toxic_signal_to_amplify():
    benign_boost = effective_probability(0.01, 1.0, U) - 0.01
    doubtful_boost = effective_probability(0.20, 1.0, U) - 0.20
    assert benign_boost < 0.05 < doubtful_boost


def test_expected_utilities_are_probability_weighted():
    eu0 = expected_utilities(0.0, 1.0, False, U)
    eu1 = expected_utilities(1.0, 1.0, False, U)
    assert eu0["publish"] == U.publish_benign and eu1["block"] == U.block_harmful


# ------------------------------------------------------------------ coordinación
def test_duplicates_from_many_accounts_raise_score():
    mem = ShortTermMemory()
    toks = tokenize("you pathetic idiot nobody wants you here just leave now")
    for i in range(6):
        mem.add(f"bot{i}", 100.0 + i, toks)
    f = mem.features("bot9", 110.0, toks)
    assert f["score"] >= 0.5 and f["distinct_users_similar"] >= 5


def test_unrelated_comments_do_not_raise_score():
    mem = ShortTermMemory()
    for i in range(10):
        mem.add(f"u{i}", 100.0 + i, tokenize(f"completely different sentence number {i} about topic {i * 7}"))
    f = mem.features("ux", 111.0, tokenize("great article thanks for sharing the sources"))
    assert f["score"] == 0.0


def test_short_repeated_comments_are_ignored():
    mem = ShortTermMemory()
    for i in range(8):
        mem.add(f"u{i}", 100.0 + i, tokenize("thanks"))
    assert mem.features("ux", 110.0, tokenize("thanks"))["score"] == 0.0


def test_old_comments_leave_the_window():
    mem = ShortTermMemory(window_s=60.0)
    toks = tokenize("you pathetic idiot nobody wants you here just leave now")
    for i in range(6):
        mem.add(f"bot{i}", 0.0 + i, toks)
    assert mem.features("bot9", 500.0, toks)["score"] == 0.0
