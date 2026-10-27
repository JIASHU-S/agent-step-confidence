"""Hidden states read BEFORE the action is written, by replaying recorded episodes (GPU, no generation).

    python scripts/07_preaction_features.py --run runs/qwen7b

For every recorded step t the transcript is rebuilt exactly as the agent saw it, and one teacher-forced pass over
prompt + history + step text gives the residual stream at two earlier points:
    pre_step    last token of "Thought t:"   -> before the model has reasoned about this step
    pre_action  last token before "Action t:" -> after the Thought, before the action is chosen
(The probe in 03 reads the Action tokens themselves: after the action is written, before it is executed.)

Writes <run>/feats_pre/<qid>.npz with arrays pre_step, pre_action of shape [n_steps, n_layers, d] and a flag
array has_action. Resumable: complete, valid files are skipped. --verify N recomputes the Action-span mean for the first
N episodes and prints its cosine similarity with the stored 03 features (should be ~1.0).
"""
import argparse
import json
import re
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.data import load_hotpot  # noqa: E402
from agentconf.checkpoints import check_config, run_lock  # noqa: E402
from agentconf.io import save_npz  # noqa: E402
from agentconf.model import load_model  # noqa: E402
from agentconf.prompts import react_messages, render  # noqa: E402
from agentconf.runs import load_run  # noqa: E402


def token_at(offsets, char):
    """Index of the token covering character `char` (or the last token starting before it)."""
    best = 0
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if a <= char < b:
            return i
        if a <= char:
            best = i
    return best


def valid_checkpoint(path, n_steps, n_layers):
    """Only skip complete, finite replay features for all recorded steps."""
    try:
        with np.load(path, allow_pickle=False) as data:
            before, action, flags = data["pre_step"], data["pre_action"], data["has_action"]
            return (before.shape == action.shape and before.ndim == 3
                    and before.shape[:2] == (n_steps, n_layers) and before.shape[2] > 0
                    and flags.shape == (n_steps,) and flags.dtype == np.bool_
                    and np.isfinite(before).all() and np.isfinite(action).all())
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
        return False


@torch.no_grad()
def step_states(model, tok, full, chars, layers):
    """Hidden states [len(chars), n_layers, d] at the tokens covering the given character positions."""
    enc = tok(full, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True,
              return_token_type_ids=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    idx = [token_at(offsets, c) for c in chars]
    enc = {k: v.to(model.device) for k, v in enc.items()}
    hs = model(**enc, output_hidden_states=True, logits_to_keep=1, use_cache=False).hidden_states
    out = torch.stack([torch.stack([hs[l][0, i, :] for l in layers]) for i in idx]).float().cpu().numpy()
    return out, offsets, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--verify", type=int, default=3)
    args = ap.parse_args()
    with run_lock(args.run):
        collect(args)


def collect(args):
    meta = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))
    if "no_4bit" in meta:
        check_config(meta, {"no_4bit": args.no_4bit})
    layers = meta["layers"]
    _, episodes, steps = load_run(args.run)
    by_q = {}
    for s in steps:
        by_q.setdefault(s["qid"], []).append(s)
    examples = {ex["id"]: ex for ex in load_hotpot(meta["n_train"] + meta["n_test"], seed=meta["seed"])}
    missing = [e["qid"] for e in episodes if e["qid"] not in examples]
    if missing:
        raise SystemExit(f"{len(missing)} recorded questions not found in the reloaded split, e.g. {missing[:3]}")

    out_dir = args.run / "feats_pre"
    out_dir.mkdir(exist_ok=True)
    todo = [e for e in episodes
            if not valid_checkpoint(out_dir / f"{e['qid']}.npz", e["n_steps"], len(layers))]
    print(f"{len(episodes) - len(todo)} done, {len(todo)} to replay", flush=True)
    if not todo:
        return
    model, tok = load_model(meta["model"], load_in_4bit=not args.no_4bit)
    t0 = time.time()
    for n, ep in enumerate(todo, 1):
        ex = examples[ep["qid"]]
        prompt = render(tok, react_messages(ex))
        traj, pre_step, pre_action, has_action = "", [], [], []
        for s in sorted(by_q[ep["qid"]], key=lambda s: s["t"]):
            t, text = s["t"], s["text"]
            base = len(prompt + traj)
            head = f"Thought {t}:"
            if not text.startswith(head):
                raise SystemExit(f"{ep['qid']} step {t}: recorded text does not start with {head!r}")
            m = re.search(r"Action\s*\d*\s*:", text)
            has = bool(m and s["kind"] != "invalid")
            a0 = m.start() if has else len(text)
            j = a0 - 1
            while j > 0 and text[j].isspace():
                j -= 1
            full = prompt + traj + text
            states, offsets, hs = step_states(model, tok, full, [base + len(head) - 1, base + j], layers)
            pre_step.append(states[0])
            pre_action.append(states[1])
            has_action.append(has)
            if n <= args.verify:
                with np.load(args.run / "feats" / f"{ep['qid']}.npz", allow_pickle=False) as saved:
                    stored = saved["mean"][t - 1].astype(np.float32)
                span = [i for i, (a, b) in enumerate(offsets) if b > base + (a0 if has else 0) and b > a]
                mean = torch.stack([hs[l][0, span, :].mean(0) for l in layers]).float().cpu().numpy()
                cos = [float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
                       for u, v in zip(mean, stored)]
                print(f"verify {ep['qid']} t={t}: cosine with stored action features min={min(cos):.4f}", flush=True)
            traj = traj + text + "\n" + f"Observation {t}: {s['obs']}\n"
        save_npz(out_dir / f"{ep['qid']}.npz", pre_step=np.stack(pre_step).astype(np.float16),
                 pre_action=np.stack(pre_action).astype(np.float16), has_action=np.array(has_action))
        if n % 25 == 0 or n == len(todo):
            rate = (time.time() - t0) / n
            print(f"[{n}/{len(todo)}] {rate:.1f}s/episode, ~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
