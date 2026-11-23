"""Held-out selection rule (agentconf/heldout.py)."""
import pytest

from agentconf.heldout import hash_order, max_cosine, select_heldout

POOL = [("old1", "Which band released the album Ok Computer?"),
        ("a", "Who directed the film that won the 1998 Palme d'Or?"),
        ("b", "who directed the film that won the 1998 palme dor"),  # same text as a after normalisation
        ("c", "What is the capital of the country where the Danube ends?"),
        ("d", "In which year was the founder of Honda born?"),
        ("e", "Which river flows through the city where Mozart was born, Austria?"),
        ("f", "How many players are on a water polo team in the pool?"),
        ("g", "Which writer created the detective Hercule Poirot and Miss Marple?")]
REFS = ["What is the capital of the country where the Danube ends?",  # exact match of c
        "Which river flows through the city where Mozart was born in Austria?",  # near duplicate of e
        "Name the largest moon of Saturn."]


def test_exclusions_and_sizes():
    (cal, fin), rep = select_heldout(POOL, {"old1"}, REFS, [1, 2])
    chosen = set(cal + fin)
    assert not chosen & {"old1", "c", "e"}
    assert len({"a", "b"} & chosen) <= 1
    assert rep == {"pool": 8, "legacy_id": 1, "pool_text_repeat": 1, "reference_exact": 1, "reference_near": 1,
                   "eligible": 4}
    eligible = [i for i in hash_order([i for i, _ in POOL]) if i in {"a", "b", "d", "f", "g"}]
    assert cal + fin == [i for i in eligible if i in chosen][:3]


def test_deterministic_and_order_independent():
    s1, _ = select_heldout(POOL, {"old1"}, REFS, [2, 2])
    s2, _ = select_heldout(POOL[::-1], {"old1"}, REFS[::-1], [2, 2])
    assert s1 == s2


def test_too_few_eligible():
    with pytest.raises(ValueError, match="eligible"):
        select_heldout(POOL, {"old1"}, REFS, [3, 3])


def test_max_cosine():
    c = max_cosine(["cat sat on mat", "quantum field theory"], ["cat sat on mat", "dogs bark"])
    assert c[0] == pytest.approx(1) and c[1] == 0
