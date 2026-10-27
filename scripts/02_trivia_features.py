"""Source domain for RQ2: closed-book TriviaQA answers, their correctness, and hidden states over the answer span.

    python scripts/02_trivia_features.py --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen7b --n 2000
Uses the same layer indices as the agent run in --out.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.checkpoints import check_config, read_meta, recover_trivia, run_lock, seed_question  # noqa: E402
from agentconf.data import load_trivia  # noqa: E402
from agentconf.io import append_jsonl, save_npz, write_json, write_jsonl  # noqa: E402
from agentconf.model import generate, load_model, span_features  # noqa: E402
from agentconf.prompts import qa_messages, render  # noqa: E402
from agentconf.text import matches_alias  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("n must be positive")
    with run_lock(args.out):
        collect(args)


def collect(args):
    out = args.out
    agent_meta = read_meta(out / "meta.json")
    check_config(agent_meta, {k: v for k, v in {"model": args.model, "no_4bit": args.no_4bit}.items()
                              if k in agent_meta})
    layers = agent_meta["layers"]
    meta_path = out / "trivia_meta.json"
    config = {k: v for k, v in vars(args).items() if k != "out"}
    config.update(layers=layers, sampling_seed="per_question_v1",
                  question_selection="unique_id_before_shuffle_v1")
    saved = read_meta(meta_path) if meta_path.exists() else None
    if saved is not None:
        check_config(saved, config)
    elif any((out / p).exists() for p in ("trivia.jsonl", "trivia_feats.npz", "trivia.progress.jsonl")):
        raise ValueError("Existing TriviaQA outputs have no trivia_meta.json; preserve them before starting a new collection")
    examples = load_trivia(args.n, seed=args.seed)
    if len(examples) != args.n or len({ex["id"] for ex in examples}) != args.n:
        raise ValueError("TriviaQA must contain the requested number of unique question IDs")
    ids = [ex["id"] for ex in examples]
    if saved is not None:
        check_config(saved, {"ids": ids})
    meta = {**config, "ids": ids, "complete": False}
    # Consumers must not read an old or partially published aggregate during recovery.
    write_json(meta_path, meta)
    rows = recover_trivia(out, layers, ids)
    todo = [ex for ex in examples if ex["id"] not in rows]
    print(f"{len(rows)} TriviaQA questions done, {len(todo)} to run", flush=True)
    model, tok = load_model(args.model, load_in_4bit=not args.no_4bit) if todo else (None, None)
    t0 = time.time()
    for i, ex in enumerate(todo, 1):
        seed_question(args.seed, ex["id"])
        prompt = render(tok, qa_messages(ex["question"]))
        ans = generate(model, tok, prompt, max_new_tokens=24)[0].split("\n")[0].strip() or "unknown"
        f = span_features(model, tok, prompt + ans, (len(prompt), len(prompt) + len(ans)), layers)
        row = {"id": ex["id"], "question": ex["question"], "answer": ans,
               "correct": int(matches_alias(ans, ex["answers"])), "lp_mean": f["lp_mean"]}
        save_npz(out / "trivia_checkpoints" / f"{ex['id']}.npz", mean=f["mean"], last=f["last"])
        append_jsonl(out / "trivia.progress.jsonl", [row])  # commit only after the feature file is durable
        rows[ex["id"]] = row
        if i == 1 or i % 10 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            print(f"[{len(rows)}/{args.n}] {ex['id']} accuracy so far "
                  f"{np.mean([r['correct'] for r in rows.values()]):.3f} "
                  f"({rate:.1f}s/question, ~{rate * (len(todo) - i) / 3600:.1f}h left)", flush=True)

    ordered = [rows[qid] for qid in ids]
    fm, fl = [], []
    for qid in ids:
        with np.load(out / "trivia_checkpoints" / f"{qid}.npz", allow_pickle=False) as f:
            fm.append(f["mean"])
            fl.append(f["last"])
    save_npz(out / "trivia_feats.npz", mean=np.stack(fm), last=np.stack(fl))
    write_jsonl(out / "trivia.jsonl", ordered)
    write_json(meta_path, {**meta, "complete": True})
    print(f"TriviaQA export complete: {len(ordered)} questions", flush=True)


if __name__ == "__main__":
    main()
