"""Question-level commits, recovery and exclusive access for long collection runs."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import warnings
import zipfile
import zlib

import numpy as np

from .io import recover_jsonl, write_jsonl


@contextmanager
def run_lock(out):
    """OS locks are released even if the process is killed; the lock file may remain."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".collection.lock").open("a+b") as f:
        if f.seek(0, 2) == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another collection process is using {out}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def check_config(saved, current):
    """Check only specified invariant fields, before touching existing run data."""
    differences = [f"{k}: saved={saved.get(k)!r}, requested={v!r}"
                   for k, v in current.items() if saved.get(k) != v]
    if differences:
        raise ValueError("Cannot resume with different settings; use a new output directory. " + "; ".join(differences))


def read_meta(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_question(seed, qid):
    """Sampling for one question is independent of how many earlier questions were skipped."""
    from transformers import set_seed

    value = int.from_bytes(hashlib.sha256(f"{seed}:{qid}".encode()).digest()[:4], "big")
    set_seed(value)


def valid_features(path, prefix):
    try:
        with np.load(path, allow_pickle=False) as f:
            mean, last = f["mean"], f["last"]
            return (mean.shape == last.shape and mean.ndim == len(prefix) + 1
                    and mean.shape[:-1] == tuple(prefix) and mean.shape[-1] > 0
                    and np.isfinite(mean).all() and np.isfinite(last).all())
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
        return False


def recover_agent(out, layers, expected_ids):
    out = Path(out)
    episodes = {e["qid"]: e for e in recover_jsonl(out / "episodes.jsonl")}
    if episodes.keys() - set(expected_ids):
        raise ValueError("Saved episodes contain question IDs outside the requested dataset")
    steps = {}
    for s in recover_jsonl(out / "steps.jsonl"):
        steps[(s["qid"], s["t"])] = s
    kept, kept_steps = [], []
    for qid, ep in episodes.items():
        n = ep["n_steps"]
        rows = [steps[(qid, t)] for t in range(1, n + 1) if (qid, t) in steps]
        if n > 0 and len(rows) == n and valid_features(out / "feats" / f"{qid}.npz", (n, len(layers))):
            kept.append(ep)
            kept_steps.extend(rows)
        else:
            warnings.warn(f"Re-running {qid}: missing or damaged steps/features", RuntimeWarning)
    # Remove invalid completion markers first. A crash during compaction remains recoverable.
    write_jsonl(out / "episodes.jsonl", kept)
    write_jsonl(out / "steps.jsonl", kept_steps)
    return {ep["qid"] for ep in kept}


def recover_trivia(out, layers, expected_ids):
    out = Path(out)
    rows = {r["id"]: r for r in recover_jsonl(out / "trivia.progress.jsonl")}
    if rows.keys() - set(expected_ids):
        raise ValueError("Saved TriviaQA records contain unexpected question IDs")
    kept = {}
    for qid in expected_ids:
        if qid not in rows:
            continue
        if valid_features(out / "trivia_checkpoints" / f"{qid}.npz", (len(layers),)):
            kept[qid] = rows[qid]
        else:
            warnings.warn(f"Re-running TriviaQA {qid}: missing or damaged features", RuntimeWarning)
    write_jsonl(out / "trivia.progress.jsonl", kept.values())
    return kept
