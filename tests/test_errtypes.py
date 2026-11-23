import numpy as np

from agentconf.errtypes import auroc_pairs, error_types
from agentconf.metrics import auroc


def _s(qid, t, kind, correct, title=None):
    return {"qid": qid, "t": t, "kind": kind, "correct": correct, "title": title}


def test_error_types():
    steps = [
        _s("a", 1, "Search", 1, "Gold A"),
        _s("a", 2, "Search", 0, "Gold A"),       # same gold paragraph again
        _s("a", 3, "Search", 0, "Distractor"),   # new, not supporting
        _s("a", 4, "Search", 0, "Distractor"),   # distractor seen before -> repeat
        _s("a", 5, "Search", 0, None),           # nothing returned
        _s("a", 6, "Finish", 0),
        _s("b", 1, "Search", 0, "Gold A"),       # history is per question: not a repeat here
        _s("b", 2, "invalid", 0),
        _s("b", 3, "Finish", 1),
    ]
    assert list(error_types(steps)) == [
        "search_ok", "search_repeat", "search_distractor", "search_repeat", "search_nomatch", "finish_wrong",
        "search_distractor", "invalid", "finish_ok"]


def test_auroc_pairs():
    rng = np.random.default_rng(0)
    score, y = rng.random(200), rng.random(200) < 0.6
    assert np.isclose(auroc_pairs(score, y), auroc(score, y))
    # A score that is pure position is perfect pooled but at chance within each position.
    t = rng.integers(1, 4, 200)
    y = rng.random(200) < np.where(t == 1, 0.9, 0.3)
    assert auroc_pairs(-t, y) > 0.7
    assert np.isclose(auroc_pairs(-t, y, strata=t), 0.5)
    assert np.isnan(auroc_pairs(score[:3], np.ones(3, bool)))
