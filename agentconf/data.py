"""Dataset loading. HotpotQA distractor (agent episodes) and TriviaQA no-context (source domain for RQ2)."""
import random


def _load(names, config, split):
    from datasets import load_dataset

    last = None
    for name in names:
        try:
            return load_dataset(name, config, split=split)
        except Exception as e:  # noqa: BLE001 - try the next mirror name
            last = e
    raise last


def load_hotpot(n, seed=0, split="validation"):
    ds = _load(["hotpotqa/hotpot_qa", "hotpot_qa"], "distractor", split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        out.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "type": r["type"],
            "level": r["level"],
            "titles": list(r["context"]["title"]),
            "sentences": [list(s) for s in r["context"]["sentences"]],
            "supporting_titles": sorted(set(r["supporting_facts"]["title"])),
        })
    return out


def load_trivia(n, seed=0, split="validation"):
    """Sample unique questions, not duplicated evidence rows in rc.nocontext."""
    ds = _load(["mandarjoshi/trivia_qa", "trivia_qa"], "rc.nocontext", split)
    first_by_id = {}
    for i, qid in enumerate(ds["question_id"]):
        first_by_id.setdefault(qid, i)
    if n < 0 or n > len(first_by_id):
        raise ValueError(f"Requested {n} TriviaQA questions, but {split} has {len(first_by_id)} unique IDs")
    idx = list(first_by_id.values())
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        out.append({"id": r["question_id"], "question": r["question"],
                    "answers": sorted({r["answer"]["value"], *r["answer"]["aliases"]})})
    return out


def _pairs(context):
    """2Wiki context as [(title, [sentences])] from list-of-pairs, dict-of-lists or JSON-string encodings."""
    import json

    if isinstance(context, str):
        context = json.loads(context)
    if isinstance(context, dict):
        return list(zip(context["title"], context["sentences"]))
    return [(c[0], list(c[1])) for c in context]


def load_2wiki_parquet(path):
    """2WikiMultihopQA rows in the HotpotQA schema used by HotpotEnv. `supporting_ordered` keeps the order of the
    supporting facts (first hop first); `supporting_titles` is the sorted set as for HotpotQA."""
    import json

    import pandas as pd

    out = []
    for r in pd.read_parquet(path).to_dict("records"):
        ctx = _pairs(r["context"])
        sf = r["supporting_facts"]
        if isinstance(sf, str):
            sf = json.loads(sf)
        sf_titles = list(sf["title"]) if isinstance(sf, dict) else [s[0] for s in sf]
        ordered = list(dict.fromkeys(sf_titles))
        out.append({
            "id": str(r["_id"] if "_id" in r else r["id"]),
            "question": r["question"],
            "answer": r["answer"],
            "type": r.get("type", ""),
            "level": "",
            "titles": [t for t, _ in ctx],
            "sentences": [list(s) for _, s in ctx],
            "supporting_titles": sorted(set(ordered)),
            "supporting_ordered": ordered,
        })
    return out


def load_trivia_parquet(path):
    """Pinned TriviaQA rc.nocontext parquet: (unique questions by first occurrence, in file order)."""
    import pyarrow.parquet as pq

    out = {}
    for r in pq.read_table(path).to_pylist():
        out.setdefault(r["question_id"], {"id": r["question_id"], "question": r["question"],
                                          "answers": sorted({r["answer"]["value"], *r["answer"]["aliases"]})})
    return out


def trivia_order(unique, seed=0):
    """Same seeded order as load_trivia: the first n IDs are the questions of a run with --n n."""
    ids = list(unique)
    random.Random(seed).shuffle(ids)
    return ids


def load_hotpot_questions(path):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["question"]).column("question").to_pylist()


def _hotpot_record(r):
    return {
        "id": r["id"],
        "question": r["question"],
        "answer": r["answer"],
        "type": r["type"],
        "level": r["level"],
        "titles": list(r["context"]["title"]),
        "sentences": [list(s) for s in r["context"]["sentences"]],
        "supporting_titles": sorted(set(r["supporting_facts"]["title"])),
    }


def load_hotpot_parquet(path, ids):
    """Pinned HotpotQA distractor parquet, rows for `ids` in the given order (same schema as load_hotpot)."""
    import pyarrow.parquet as pq

    rows = {r["id"]: r for r in pq.read_table(path).to_pylist()}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"{len(missing)} manifest IDs are not in {path}")
    return [_hotpot_record(rows[i]) for i in ids]
