import numpy as np
import pytest

from agentconf.agent import _cut_step, _parse_score
from agentconf.env import HotpotEnv, parse_action
from agentconf.seqmodel import handoff_curve, trajectory_confidences

EX = {
    "id": "q1", "question": "Which was founded first, Kyoto University or Tohoku University?",
    "answer": "Kyoto University", "type": "comparison", "level": "easy",
    "titles": ["Kyoto University", "Tohoku University", "Osaka University"],
    "sentences": [["Kyoto University is in Kyoto.", " It was founded in 1897."],
                  ["Tohoku University is in Sendai.", " It was established in 1907."],
                  ["Osaka University is in Suita."]],
    "supporting_titles": ["Kyoto University", "Tohoku University"],
}


def test_parse_action():
    assert parse_action("Thought 1: x\nAction 1: Search[Kyoto University]\nObs")[:2] == ("Search", "Kyoto University")
    assert parse_action("Action: finish[ 1897 ]")[:2] == ("Finish", "1897")
    assert parse_action("Thought 1: hmm")[0] == "invalid"


def test_cut_step():
    text, kind, arg = _cut_step("Thought 2: next.\nAction 2: Search[Tohoku]\nObservation 2: junk")
    assert text.endswith("Search[Tohoku]") and kind == "Search" and arg == "Tohoku"
    text, kind, _ = _cut_step("Thought 2: rambling\nObservation 2: junk")
    assert kind == "invalid" and "Observation" not in text


def test_env_labels():
    env = HotpotEnv(EX)
    obs, lab = env.step("Search", "Kyoto University")
    assert obs.startswith("[Kyoto University]") and lab["correct"] == 1
    _, lab = env.step("Search", "kyoto university")  # same gold paragraph again: no new evidence
    assert lab["correct"] == 0
    _, lab = env.step("Search", "university in Sendai established 1907")  # BM25 path
    assert lab["title"] == "Tohoku University" and lab["correct"] == 1
    _, lab = env.step("Search", "Osaka University")  # distractor
    assert lab["correct"] == 0
    _, lab = env.step("Finish", "Kyoto University")
    assert lab["correct"] == 1 and env.done
    assert env.step("invalid", "")[1]["correct"] == 0


def test_canonical_groups_equivalent_searches():
    env = HotpotEnv(EX)
    assert env.canonical("Search", "Kyoto University") == env.canonical("Search", "founded 1897 Kyoto")
    assert env.canonical("Finish", "The Kyoto University") == env.canonical("Finish", "kyoto university")
    assert env.canonical("Search", "zzz") == "S:<none>"


def test_parse_score():
    assert _parse_score(" 85") == 0.85 and _parse_score("150%") == 1.0 and np.isnan(_parse_score("high"))


@pytest.mark.parametrize("text, expected", [
    ("I would rate the confidence of Action 2 as 85%.", 0.85),
    ("Step 3 follows Observation 2; confidence: 90.", 0.90),
    ("THOUGHT 1 and ACTION 2 are correct. Confidence: 95.5", 0.955),
    ("Action 2: 0", 0.0),
])
def test_parse_score_skips_step_references(text, expected):
    assert _parse_score(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "I would rate the confidence of",
    "I would rate the confidence level",
    "Action 2 follows step 1.",
])
def test_parse_score_requires_a_score(text):
    assert np.isnan(_parse_score(text))


def test_forward_recursion_and_handoff():
    S = 50
    a = np.zeros((S, 3)); b = np.ones((S, 3)); g = np.full(S, -3.0)
    steps = [{"kind": "Search", "s": 2.0}, {"kind": "Search", "s": -2.0}, {"kind": "Finish", "s": 1.0}]
    c = trajectory_confidences(steps, (a, b, np.zeros(S)), (a, b, g))
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    assert np.isclose(c["M0_product"], sig(2) * sig(-2) * sig(1))
    q1 = sig(2.0)
    q2 = q1 * sig(-2) + (1 - q1) * sig(-5)
    q3 = q2 * sig(1) + (1 - q2) * sig(-2)
    assert np.allclose(c["running"], [q1, q2, q3])
    eps = [{"running": [0.9, 0.2], "success": 0}, {"running": [0.9, 0.8], "success": 1}]
    cur = {r["tau"]: r for r in handoff_curve(eps, [0.1, 0.5])}
    assert cur[0.1]["coverage"] == 1.0 and cur[0.5]["coverage"] == 0.5
    assert cur[0.5]["kept_accuracy"] == 1.0 and cur[0.5]["mean_handoff_step"] == 2
