"""Loading a run directory written by scripts/01_run_agent.py."""
import json
from pathlib import Path

import numpy as np

from .io import read_jsonl


def load_run(out):
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text())
    episodes = {e["qid"]: e for e in read_jsonl(out / "episodes.jsonl")}
    # A crash mid-episode can leave steps of an unfinished (and later re-run) episode: keep the last copy of each
    # (qid, t) whose episode completed.
    steps = {}
    for s in read_jsonl(out / "steps.jsonl"):
        if s["qid"] in episodes:
            steps[(s["qid"], s["t"])] = s
    steps = [steps[k] for k in sorted(steps)]
    steps = [s for s in steps if s["t"] <= episodes[s["qid"]]["n_steps"]]
    return meta, list(episodes.values()), steps


def load_features(out, steps, pooling="mean"):
    """Returns [n_steps, n_layers, d] float32 aligned with `steps`."""
    out = Path(out)
    cache, rows = {}, []
    for s in steps:
        if s["qid"] not in cache:
            cache[s["qid"]] = np.load(out / "feats" / f"{s['qid']}.npz")[pooling]
        rows.append(cache[s["qid"]][s["t"] - 1])
    return np.stack(rows).astype(np.float32)
