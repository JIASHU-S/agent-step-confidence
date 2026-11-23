"""Source tasks for the source-swap experiment (protocols/source_swap_v1.md): one model-generated decision per item,
its automatic label, and hidden states over the same kind of span the agent features use.

    # weight check first: recompute stored agent features with the loaded weights (must match before anything else)
    python scripts/19_source_features.py --run runs/qwen7b --verify-agent 10

    python scripts/19_source_features.py --run runs/qwen7b --source trivia_react   --role pilot --n 50
    python scripts/19_source_features.py --run runs/qwen7b --source wiki_search    --role pilot --n 50 --wiki data/source_swap/2wiki/<rev>/dev.parquet
    python scripts/19_source_features.py --run runs/qwen7b --source wiki_search_qa --role pilot --n 50 --wiki ...

Sources
    trivia_react    C1: TriviaQA closed-book in ReAct format. Prefix ends with a fixed thought and "Action 1: Finish[";
                    the model writes the answer. Main role = the same question IDs as the run's TriviaQA (S0); pilot =
                    the next IDs in the same seeded order. Label: alias match, as for S0. Span: "Action 1: Finish[...]".
    wiki_search     S2: one ReAct step on a 2WikiMultihopQA question. Hop 1, or hop 2 after a templated first step that
                    reads the first supporting paragraph. Label and error type: the HotpotQA environment's rule.
                    Span: the Action, as in the agent run.
    wiki_search_qa  S2-QA: the same decision in QA format ("which document should be read next?"); the reply is
                    resolved to a paragraph like a Search. Span: the reply.

Outputs: <run>/sources/<source>-<role>/ with meta.json, progress.jsonl, checkpoints/<id>.npz, and on completion
items.jsonl and feats.npz (mean/last [n, layers, d]). Resumable: rerun the same command. Verification writes
<run>/sources/verify_agent.json.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.agent import STOP, _cut_step  # noqa: E402
from agentconf.checkpoints import check_config, read_meta, run_lock, seed_question, valid_features  # noqa: E402
from agentconf.data import (load_2wiki_parquet, load_hotpot_questions, load_trivia_parquet,  # noqa: E402
                            trivia_order)
from agentconf.env import parse_action  # noqa: E402
from agentconf.io import append_jsonl, read_jsonl, recover_jsonl, save_npz, write_json, write_jsonl  # noqa: E402
from agentconf.prompts import (REACT_NO_DOCS_THOUGHT, lookup_messages, react_messages,  # noqa: E402
                               react_no_docs_messages, render)
from agentconf.sources import (SOURCES, clean_title_answer, cut_finish_arg, hop_of, lookup_env,  # noqa: E402
                               order_2wiki, search_type, select_2wiki)
from agentconf.text import matches_alias  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_paths():
    """Pinned, hash-verified files recorded by the P0 dataset check (scripts/14_p0_datasets.py)."""
    specs = read_meta(ROOT / "manifests/submission_v2/p0/dataset_verification.json")["datasets"]
    return {name: ROOT / specs[name]["local_path"] for name in ("hotpotqa", "triviaqa")}


def build_items(args, run):
    paths = dataset_paths()
    if args.source == "trivia_react":
        unique = load_trivia_parquet(paths["triviaqa"])
        tm = read_meta(run / "trivia_meta.json")
        order = trivia_order(unique, tm["seed"])
        if order[: tm["n"]] != tm["ids"]:
            raise SystemExit("TriviaQA order does not reproduce the run's S0 question IDs")
        ids = tm["ids"][: args.n] if args.role == "main" else order[tm["n"]: tm["n"] + args.n]
        if len(ids) < args.n:
            raise SystemExit(f"only {len(ids)} TriviaQA questions available")
        return [{"id": i, "hop": 1, "ex": unique[i]} for i in ids], {"triviaqa_sha256": sha256(paths["triviaqa"])}
    if args.wiki is None:
        raise SystemExit("--wiki <2WikiMultihopQA parquet> is required for wiki sources")
    ordered = order_2wiki(load_2wiki_parquet(args.wiki), load_hotpot_questions(paths["hotpotqa"]))
    picked = select_2wiki(ordered, args.source, args.role, args.n)
    return ([{"id": ex["id"], "hop": hop_of(ex["id"]), "ex": ex} for ex in picked],
            {"wiki_file": str(args.wiki), "wiki_sha256": sha256(args.wiki), "wiki_usable": len(ordered),
             "label_env": "FoldedEnv (diacritic-insensitive matching)"})


def action_span(step_text, kind):
    m = re.search(r"Action\s*\d*\s*:", step_text)
    return m.start() if (m and kind != "invalid") else 0


def run_item(model, tok, source, item, layers):
    from agentconf.model import generate, span_features

    ex, hop = item["ex"], item["hop"]
    if source == "trivia_react":
        prompt = render(tok, react_no_docs_messages(ex["question"]))
        raw = generate(model, tok, prompt + REACT_NO_DOCS_THOUGHT, max_new_tokens=24, stop_strings=["]", "\n"])[0]
        ans = cut_finish_arg(raw) or "unknown"
        step_text = REACT_NO_DOCS_THOUGHT + ans + "]"
        a0 = action_span(step_text, "Finish")
        f = span_features(model, tok, prompt + step_text, (len(prompt) + a0, len(prompt) + len(step_text)), layers)
        lab = {"kind": "Finish", "arg": ans, "correct": int(matches_alias(ans, ex["answers"])),
               "etype": None, "text": step_text}
    elif source == "wiki_search":
        env, hist, _ = lookup_env(ex, hop)
        prompt = render(tok, react_messages(ex))
        t = hop
        prefix = prompt + hist + f"Thought {t}:"
        raw = generate(model, tok, prefix, max_new_tokens=128, stop_strings=STOP)[0]
        step_text, kind, arg = _cut_step(f"Thought {t}:" + raw)
        base = len(prompt + hist)
        a0 = action_span(step_text, kind)
        f = span_features(model, tok, prompt + hist + step_text, (base + a0, base + len(step_text)), layers)
        _, lab = search_type(env, kind, arg)
        lab = {"kind": kind, "arg": arg, "text": step_text, **lab}
    else:  # wiki_search_qa
        env, _, read = lookup_env(ex, hop)
        prompt = render(tok, lookup_messages(ex, read))
        raw = generate(model, tok, prompt, max_new_tokens=32)[0]
        ans = clean_title_answer(raw) or "unknown"
        f = span_features(model, tok, prompt + ans, (len(prompt), len(prompt) + len(ans)), layers)
        _, lab = search_type(env, "Search", ans)
        lab = {"kind": "Search", "arg": ans, "text": raw.split("\n")[0], **lab}
    row = {"id": item["id"], "source": source, "hop": hop, "question": ex["question"], **lab,
           "lp_mean": f["lp_mean"], "n_tokens": f["n_tokens"]}
    return row, f


def collect(args):
    run = args.run
    agent_meta = read_meta(run / "meta.json")
    if args.model != agent_meta["model"]:
        raise SystemExit(f"--model {args.model} differs from the agent run's {agent_meta['model']}")
    layers = agent_meta["layers"]
    out = run / "sources" / f"{args.source}-{args.role}"
    items, data_info = build_items(args, run)
    ids = [it["id"] for it in items]
    config = {"model": args.model, "revision": args.revision, "no_4bit": args.no_4bit, "source": args.source,
              "role": args.role, "n": args.n, "seed": args.seed, "layers": layers, "protocol": "source_swap_v1",
              **data_info}
    meta_path = out / "meta.json"
    if meta_path.exists():
        saved = read_meta(meta_path)
        check_config(saved, {k: v for k, v in config.items() if k != "wiki_file"})
        check_config(saved, {"ids": ids})
    out.mkdir(parents=True, exist_ok=True)
    meta = {**config, "ids": ids, "complete": False}
    write_json(meta_path, meta)

    done = {}
    for r in recover_jsonl(out / "progress.jsonl"):
        if r["id"] in set(ids) and valid_features(out / "checkpoints" / f"{r['id']}.npz", (len(layers),)):
            done[r["id"]] = r
    write_jsonl(out / "progress.jsonl", done.values())
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.source}-{args.role}: {len(done)} done, {len(todo)} to run", flush=True)
    if todo:
        from agentconf.model import load_model

        model, tok = load_model(args.model, load_in_4bit=not args.no_4bit, revision=args.revision)
        meta["resolved_revision"] = getattr(model.config, "_commit_hash", None)
        write_json(meta_path, meta)
        t0 = time.time()
        for i, it in enumerate(todo, 1):
            seed_question(args.seed, it["id"])
            row, f = run_item(model, tok, args.source, it, layers)
            save_npz(out / "checkpoints" / f"{it['id']}.npz", mean=f["mean"], last=f["last"])
            append_jsonl(out / "progress.jsonl", [row])
            done[it["id"]] = row
            if i == 1 or i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                acc = np.mean([r["correct"] for r in done.values()])
                print(f"[{len(done)}/{len(ids)}] acc {acc:.3f} ({rate:.2f}s/item, ~{rate * (len(todo) - i) / 60:.0f} min left)",
                      flush=True)
        meta["seconds_per_item"] = (time.time() - t0) / len(todo)

    rows = [done[i] for i in ids]
    fm, fl = [], []
    for i in ids:
        with np.load(out / "checkpoints" / f"{i}.npz", allow_pickle=False) as f:
            fm.append(f["mean"])
            fl.append(f["last"])
    save_npz(out / "feats.npz", mean=np.stack(fm), last=np.stack(fl))
    write_jsonl(out / "items.jsonl", rows)
    write_json(meta_path, {**meta, "complete": True})
    summarize(rows, meta)


def summarize(rows, meta):
    y = np.array([r["correct"] for r in rows])
    kinds = {}
    for r in rows:
        key = r["etype"] or ("correct" if r["correct"] else "wrong")
        kinds[key] = kinds.get(key, 0) + 1
    by_hop = {h: float(np.mean([r["correct"] for r in rows if r["hop"] == h]))
              for h in sorted({r["hop"] for r in rows})}
    print(json.dumps({"n": len(rows), "accuracy": float(y.mean()), "negatives": int((y == 0).sum()),
                      "accuracy_by_hop": by_hop, "types": kinds,
                      "kinds": {k: sum(r["kind"] == k for r in rows) for k in {r["kind"] for r in rows}},
                      "seconds_per_item": meta.get("seconds_per_item")}, indent=1))


def verify_agent(args):
    """Recompute action-span features of stored agent steps with the loaded weights and compare."""
    from agentconf.model import load_model, span_features

    run = args.run
    agent_meta = read_meta(run / "meta.json")
    layers = agent_meta["layers"]
    pilot = {r["id"]: r for r in read_jsonl(ROOT / "manifests/submission_v2/p0/pilot_examples.jsonl")}
    steps = {}
    for s in read_jsonl(run / "steps.jsonl"):
        steps.setdefault(s["qid"], {})[s["t"]] = s
    qids = [q for q in pilot if q in steps][: args.verify_agent]
    model, tok = load_model(agent_meta["model"], load_in_4bit=not agent_meta.get("no_4bit", False),
                            revision=args.revision)
    report = {"model": agent_meta["model"], "requested_revision": args.revision,
              "resolved_revision": getattr(model.config, "_commit_hash", None), "steps": []}
    for q in qids:
        ex = pilot[q]
        prompt = render(tok, react_messages(ex))
        traj = ""
        with np.load(run / "feats" / f"{q}.npz", allow_pickle=False) as saved:
            stored_mean, stored_last = saved["mean"].astype(np.float32), saved["last"].astype(np.float32)
        for t in sorted(steps[q]):
            s = steps[q][t]
            kind = parse_action(s["text"])[0]
            a0 = action_span(s["text"], kind)
            base = len(prompt + traj)
            f = span_features(model, tok, prompt + traj + s["text"], (base + a0, base + len(s["text"])), layers)
            cos = [float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
                   for u, v in zip(f["mean"].astype(np.float32), stored_mean[t - 1])]
            rel = float(np.max(np.abs(f["last"].astype(np.float32) - stored_last[t - 1]))
                        / (np.max(np.abs(stored_last[t - 1])) + 1e-9))
            report["steps"].append({"qid": q, "t": t, "min_cosine_mean": min(cos), "max_rel_diff_last": rel,
                                    "lp_mean_new": f["lp_mean"], "lp_mean_stored": s["lp_mean"]})
            traj = traj + s["text"] + "\n" + f"Observation {t}: {s['obs']}\n"
    cmin = min(r["min_cosine_mean"] for r in report["steps"])
    dlp = max(abs(r["lp_mean_new"] - r["lp_mean_stored"]) for r in report["steps"])
    report.update({"n_steps": len(report["steps"]), "min_cosine": cmin, "max_abs_lp_mean_diff": dlp,
                   "pass": bool(cmin >= args.verify_min_cosine)})
    write_json(run / "sources" / "verify_agent.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=1))
    if not report["pass"]:
        raise SystemExit("Loaded weights do not reproduce the stored agent features: do not collect sources with them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", help="defaults to the agent run's model")
    ap.add_argument("--revision", help="model repository revision (commit); recorded in meta")
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--role", choices=("pilot", "main"), default="pilot")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wiki", type=Path, help="2WikiMultihopQA parquet (dev split)")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--verify-agent", type=int, default=0, help="check N pilot episodes' stored features, then exit")
    ap.add_argument("--verify-min-cosine", type=float, default=0.999)
    args = ap.parse_args()
    (args.run / "sources").mkdir(parents=True, exist_ok=True)
    if args.verify_agent:
        verify_agent(args)
        return
    if args.source is None:
        ap.error("--source is required unless --verify-agent is given")
    args.model = args.model or read_meta(args.run / "meta.json")["model"]
    verify = args.run / "sources" / "verify_agent.json"
    if args.role == "main" and not (verify.exists() and read_meta(verify).get("pass")):
        raise SystemExit("run --verify-agent first: main collections need weights that reproduce the agent features")
    with run_lock(args.run / "sources" / f"{args.source}-{args.role}"):
        collect(args)


if __name__ == "__main__":
    main()
