import pandas as pd

from agentconf.data import load_2wiki_parquet
from agentconf.prompts import lookup_messages, render
from agentconf.sources import (PILOT_RESERVED, FoldedEnv, clean_title_answer, fold, cut_finish_arg, hop_of, lookup_env, order_2wiki,
                               search_type, select_2wiki)

EX = {
    "id": "w1", "question": "Where was the director of film Spirited Away born?", "answer": "Tokyo",
    "type": "compositional", "level": "",
    "titles": ["Spirited Away", "Hayao Miyazaki", "Studio Ghibli"],
    "sentences": [["Spirited Away is a 2001 film directed by Hayao Miyazaki."],
                  ["Hayao Miyazaki was born in Tokyo."], ["Studio Ghibli is an animation studio."]],
    "supporting_titles": ["Hayao Miyazaki", "Spirited Away"],
    "supporting_ordered": ["Spirited Away", "Hayao Miyazaki"],
}


def _ex(i, q=None):
    return {**EX, "id": f"q{i}", "question": q or f"question {i}?"}


def test_load_2wiki_parquet(tmp_path):
    import json

    # Mirrors store nested fields either as structs of lists or as JSON strings of [title, ...] pairs.
    structs = {"_id": "a1", "type": "compositional", "question": EX["question"], "answer": "Tokyo",
               "context": {"title": ["Spirited Away", "Hayao Miyazaki"], "sentences": [["s1."], ["s2."]]},
               "supporting_facts": {"title": ["Spirited Away", "Hayao Miyazaki", "Spirited Away"], "sent_id": [0, 0, 1]}}
    strings = {**structs, "_id": "a2",
               "context": json.dumps([["Spirited Away", ["s1."]], ["Hayao Miyazaki", ["s2."]]]),
               "supporting_facts": json.dumps([["Spirited Away", 0], ["Hayao Miyazaki", 0]])}
    for i, row in enumerate((structs, strings)):
        path = tmp_path / f"dev{i}.parquet"
        pd.DataFrame([row]).to_parquet(path)
        ex = load_2wiki_parquet(path)[0]
        assert ex["id"] == row["_id"] and ex["titles"] == ["Spirited Away", "Hayao Miyazaki"]
        assert ex["sentences"] == [["s1."], ["s2."]]
        assert ex["supporting_ordered"] == ["Spirited Away", "Hayao Miyazaki"]
        assert ex["supporting_titles"] == ["Hayao Miyazaki", "Spirited Away"]


def test_selection_is_disjoint_and_excludes_target():
    exs = [_ex(i) for i in range(2 * PILOT_RESERVED + 20)] + [_ex(9999, "Who directed Spirited Away")]
    ordered = order_2wiki(exs, exclude_questions=["who directed spirited away?"])
    assert all(ex["id"] != "q9999" for ex in ordered)
    a = select_2wiki(ordered, "wiki_search", "pilot", 50)
    b = select_2wiki(ordered, "wiki_search_qa", "pilot", 50)
    c = select_2wiki(ordered, "wiki_search", "main", 50)
    ids = [set(x["id"] for x in s) for s in (a, b, c)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert select_2wiki(ordered, "wiki_search", "pilot", 50) == a  # deterministic
    assert {hop_of(f"q{i}") for i in range(50)} == {1, 2}


def test_lookup_env_and_labels():
    env, hist, read = lookup_env(EX, 1)
    assert hist == "" and read == []
    _, lab = search_type(env, "Search", "Spirited Away")
    assert lab == {"correct": 1, "title": "Spirited Away", "etype": "search_ok"}

    env, hist, read = lookup_env(EX, 2)
    assert hist.startswith("Thought 1: I should look up Spirited Away.\nAction 1: Search[Spirited Away]\n")
    assert hist.endswith("\n") and "Observation 1: [Spirited Away]" in hist
    assert read[0][0] == "Spirited Away" and not read[0][1].startswith("[")
    assert search_type(env, "Search", "Spirited Away")[1]["etype"] == "search_repeat"
    assert search_type(env, "Search", "Studio Ghibli")[1]["etype"] == "search_distractor"
    assert search_type(env, "Search", "Hayao Miyazaki")[1]["etype"] == "search_ok"
    assert search_type(env, "Search", "zzz qqq")[1]["etype"] == "search_nomatch"
    assert search_type(env, "invalid", "")[1]["etype"] == "invalid"


def test_answer_cleaning_and_prompt():
    assert clean_title_answer('"Hayao Miyazaki".\nbecause') == "Hayao Miyazaki"
    assert clean_title_answer("Search[Studio Ghibli]") == "Studio Ghibli"
    assert cut_finish_arg(" Tokyo] extra") == "Tokyo"
    assert cut_finish_arg("Tokyo\nmore") == "Tokyo"

    class Tok:
        chat_template = None

    text = render(Tok(), lookup_messages(EX, [("Spirited Away", "Spirited Away is a 2001 film.")]))
    assert "Already read:\n[Spirited Away] Spirited Away is a 2001 film.\nWhich document should be read next?" in text


def test_folded_env_matches_without_accents():
    ex = {**EX, "titles": ["Ácratas", "Xawery Żuławski", "Studio Ghibli"], "supporting_titles": ["Ácratas"],
          "supporting_ordered": ["Ácratas", "Xawery Żuławski"]}
    assert fold("Żuławski Ácratas") == "Zulawski Acratas"
    env = FoldedEnv(ex)
    assert env.titles[env.resolve("Acratas")] == "Ácratas"
    assert env.titles[env.resolve("Xawery Zulawski")] == "Xawery Żuławski"
    assert search_type(env, "Search", "Acratas")[1]["etype"] == "search_ok"
    assert search_type(env, "Search", "Ácratas")[1]["etype"] == "search_repeat"


def test_balance_and_split():
    from agentconf.sources import balance_indices, source_split

    ids = [f"x{i}" for i in range(100)]
    y = [1 if i % 4 else 0 for i in range(100)]  # 75 positives, 25 negatives
    idx = balance_indices(ids, y, 20)
    assert len(idx) == 40 and sum(y[i] for i in idx) == 20
    assert balance_indices(list(reversed(ids)), list(reversed(y)), 20) == sorted(99 - i for i in idx)  # order-free
    tr = source_split(ids)
    assert sum(tr) == 80 and source_split(ids) == tr
    import pytest

    with pytest.raises(ValueError):
        balance_indices(ids, y, 30)
