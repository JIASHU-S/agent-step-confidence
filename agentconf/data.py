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
