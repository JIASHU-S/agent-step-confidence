"""Run the ReAct agent on HotpotQA (distractor) and record step labels, signals and hidden-state features.

Resumable: episodes already in <out>/episodes.jsonl are skipped.

Example (RTX 4070 Ti, 12 GB):
    python scripts/01_run_agent.py --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen7b
Quick check:
    python scripts/01_run_agent.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/smoke --n-train 5 --n-test 5 --no-4bit
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.agent import run_episode  # noqa: E402
from agentconf.checkpoints import check_config, read_meta, recover_agent, run_lock, seed_question  # noqa: E402
from agentconf.data import load_hotpot  # noqa: E402
from agentconf.io import append_jsonl, save_npz, write_json  # noqa: E402
from agentconf.model import load_model, select_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--k-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--layers", default="auto", help="'auto' (8 evenly spaced), 'all', or comma-separated indices")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-verbal", action="store_true")
    args = ap.parse_args()
    if args.n_train < 0 or args.n_test < 0 or args.n_train + args.n_test == 0 or args.max_steps < 1:
        ap.error("question counts must be nonnegative (with a positive total), and max-steps must be positive")
    with run_lock(args.out):
        collect(args)


def collect(args):
    out = args.out
    meta_path = out / "meta.json"
    saved = read_meta(meta_path) if meta_path.exists() else None
    current = {k: v for k, v in vars(args).items() if k not in ("out", "layers")}
    if saved is not None:
        check_config(saved, current)
        if "layers_spec" in saved:
            check_config(saved, {"layers_spec": args.layers})
    elif any((out / p).exists() for p in ("episodes.jsonl", "steps.jsonl")) or any((out / "feats").glob("*.npz")):
        raise ValueError("Existing agent outputs have no meta.json; restore it or use a new output directory")

    examples = load_hotpot(args.n_train + args.n_test, seed=args.seed)
    if len(examples) != args.n_train + args.n_test:
        raise ValueError("Dataset contains fewer questions than requested")
    split = {ex["id"]: ("train" if i < args.n_train else "test") for i, ex in enumerate(examples)}
    if len(split) != len(examples):
        raise ValueError("Dataset contains duplicate question IDs")
    if saved is not None:
        check_config(saved, {"split": split})

    model, tok = None, None
    # Legacy metadata stored resolved layers only. Resolve explicit legacy layer requests before recovery.
    if saved is None or ("layers_spec" not in saved and args.layers != "auto"):
        model, tok = load_model(args.model, load_in_4bit=not args.no_4bit)
        layers = select_layers(model.config.num_hidden_layers + 1, args.layers)
        if saved is not None:
            check_config(saved, {"layers": layers})
    else:
        layers = saved["layers"]
    meta = saved or {**current, "out": str(out), "layers": layers, "layers_spec": args.layers,
                     "split": split, "sampling_seed": "per_question_v1"}
    if saved is None:
        write_json(meta_path, meta)
    if meta.get("sampling_seed", "legacy") not in ("legacy", "per_question_v1"):
        raise ValueError("Unsupported sampling seed protocol in meta.json")

    done = recover_agent(out, layers, split)
    todo = [ex for ex in examples if ex["id"] not in done]
    print(f"{len(done)} episodes done, {len(todo)} to run; layers {layers}", flush=True)
    if not todo:
        return
    if model is None:
        model, tok = load_model(args.model, load_in_4bit=not args.no_4bit)
    check_config(meta, {"layers": select_layers(model.config.num_hidden_layers + 1, args.layers)})
    t0 = time.time()
    for i, ex in enumerate(todo, 1):
        if meta.get("sampling_seed") == "per_question_v1":
            seed_question(args.seed, ex["id"])
        episode_start = time.time()

        def progress(step, phase):
            print(f"[{len(done) + i}/{len(examples)} step {step}] {ex['id']} {phase} "
                  f"({time.time() - episode_start:.1f}s this episode)", flush=True)

        ep, steps, fm, fl = run_episode(model, tok, ex, layers, max_steps=args.max_steps, k_samples=args.k_samples,
                                        temperature=args.temperature, verbal=not args.no_verbal, progress=progress)
        save_npz(out / "feats" / f"{ex['id']}.npz", mean=fm, last=fl)
        ep["split"] = split[ex["id"]]
        for s in steps:
            s["split"] = ep["split"]
        append_jsonl(out / "steps.jsonl", steps)
        append_jsonl(out / "episodes.jsonl", [ep])  # written last, so a crash mid-episode reruns it
        rate = (time.time() - t0) / i
        print(f"[{len(done) + i}/{len(examples)}] {ex['id']} steps={ep['n_steps']} success={ep['success']} "
              f"({rate:.1f}s/episode, ~{rate * (len(todo) - i) / 3600:.1f}h left)", flush=True)


if __name__ == "__main__":
    main()
