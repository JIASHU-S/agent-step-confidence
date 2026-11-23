"""Source-swap confirmation, protocol v1.2 (protocols/source_swap_v1.md section 11). CPU.

Freeze once, before any held-out trajectory is generated (records the model set and their target runs):
    python scripts/22_confirm_source_swap.py --freeze
Confirm, after every frozen model's held-out agent run is complete:
    python scripts/22_confirm_source_swap.py

The confirmation always uses exactly the frozen model set. It first runs every label-free check for all models:
frozen source data, code and manifest hashes; each target run's model, revision, stored layers, manifest hash and
split; the episode set equals the manifest's final IDs exactly; every episode has steps 1..n in that split and a
readable, finite feature file of shape [n, layers, d] with the source feature width. Only when everything passes does
it score (scripts/20_source_swap.py, --readout upper for the tests and --readout selected for the v1.1 estimate)
into a staging directory; the per-model outputs are moved into place and results/source_swap_confirmation.json is
written only after every model has been scored.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.checkpoints import run_lock, valid_features  # noqa: E402
from agentconf.io import atomic_open, read_jsonl, write_json  # noqa: E402

CODE = ["scripts/01_run_agent.py", "scripts/20_source_swap.py", "scripts/21_select_heldout.py",
        "scripts/22_confirm_source_swap.py"] + [f"agentconf/{m}.py" for m in (
            "agent", "checkpoints", "data", "env", "errtypes", "heldout", "io", "metrics", "model", "probes",
            "prompts", "runs", "sources", "text")]
SOURCE_FILES = {"S0/trivia.jsonl": "trivia.jsonl", "S0/trivia_feats.npz": "trivia_feats.npz",
                "C1/items.jsonl": "sources/trivia_react-main/items.jsonl",
                "C1/feats.npz": "sources/trivia_react-main/feats.npz",
                "S2QA/items.jsonl": "sources/wiki_search_qa-main/items.jsonl",
                "S2QA/feats.npz": "sources/wiki_search_qa-main/feats.npz"}
DEFAULT_PAIRS = ["runs/qwen7b=runs/qwen7b-final", "runs/llama8b=runs/llama8b-final"]
KEY = {"I": "I", "G": "H2_distractor_S2QA_minus_S0", "C": "finish_S2QA_minus_S0"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)


def framing(per_model):
    """Pre-registered outcome -> framing map (section 11.5); a condition counts only if it holds in every model."""
    allm = lambda k: all(m[k] for m in per_model.values())  # noqa: E731
    h1, gain = allm("H1_this_model"), allm("gain_this_model")
    if h1 and gain:
        if allm("finish_noninferior_this_model"):
            return "generalization", ("retrieval source adds distractor coverage with finish loss below the margin; "
                                      "claims beyond HotpotQA need the MuSiQue confirmation")
        if allm("finish_cost_this_model"):
            return "tradeoff", "these two training sources show a trade-off between distractor and finish coverage"
        return "gain_only", "claim the distractor gain; finish cost is inconclusive or differs across models"
    if h1:
        return "interaction_only", ("interaction confirmed, retrieval gain not confirmed across models: report I, G "
                                    "and C per model, without attributing the interaction to the distractor side")
    if gain:
        return "gain_without_interaction", "report the distractor gain estimate; the interaction is not confirmed"
    return "neither_confirmed", "phase-1 layer pattern did not replicate; report as a negative result"


def freeze(args, pairs):
    if args.frozen.exists() and not args.force:
        raise SystemExit(f"{args.frozen} exists; pass --force only before any held-out trajectory exists")
    for _, target in pairs:
        run = ROOT / target
        if any((run / name).exists() for name in ("meta.json", "steps.jsonl", "episodes.jsonl", "feats")):
            raise SystemExit("held-out collection already started: freezing now would not be a pre-registration")
    out = {"protocol": "source_swap_v1.2 (section 11)", "heldout_manifest": {"path": rel(args.manifest),
                                                                           "sha256": sha256(args.manifest)},
           "split": args.split, "output": rel(args.out),
           "scoring": {"pca": args.pca, "n_boot": args.n_boot, "min_m": args.min_m},
           "models": {}, "code": {c: sha256(ROOT / c) for c in CODE}}
    for s, t in pairs:
        meta = json.loads((ROOT / s / "meta.json").read_text(encoding="utf-8"))
        rev = json.loads((ROOT / s / "sources" / "trivia_react-main" / "meta.json").read_text(encoding="utf-8"))
        out["models"][rel(ROOT / s)] = {"target_run": rel(ROOT / t), "model": meta["model"], "layers": meta["layers"],
                                        "model_revision": rev.get("resolved_revision"),
                                        "files": {k: sha256(ROOT / s / v) for k, v in SOURCE_FILES.items()}}
    args.frozen.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.frozen, out)
    print(f"froze {len(CODE)} code files, the manifest and {len(pairs)} models -> {rel(args.frozen)}")


def check_target(t, f, fz, ids, width):
    """Label-free checks of one target run; returns a list of problems."""
    t = ROOT / t
    problems = []
    if not (t / "meta.json").exists():
        return [f"{rel(t)}: no agent run yet ({len(ids)} manifest questions without an episode)"]
    tm = json.loads((t / "meta.json").read_text(encoding="utf-8"))
    for key, want in (("model", f["model"]), ("layers", f["layers"]), ("revision", f["model_revision"]),
                      ("ids_sha256", fz["heldout_manifest"]["sha256"])):
        if tm.get(key) != want:
            problems.append(f"{rel(t)}: {key} is {tm.get(key)!r}, expected {want!r}")
    if fz["split"] not in (tm.get("splits") or "").split(","):
        problems.append(f"{rel(t)}: split {fz['split']} was not run")
    eps = {e["qid"]: e for e in read_jsonl(t / "episodes.jsonl")} if (t / "episodes.jsonl").exists() else {}
    missing, extra = set(ids) - set(eps), set(eps) - set(ids)
    if missing or extra:
        problems.append(f"{rel(t)}: {len(missing)} manifest questions without an episode, {len(extra)} extra episodes")
    steps = {}
    for s in read_jsonl(t / "steps.jsonl") if (t / "steps.jsonl").exists() else []:
        steps.setdefault(s["qid"], {})[s["t"]] = s
    bad = []
    for q in set(ids) & set(eps):
        n, rows = eps[q]["n_steps"], steps.get(q, {})
        ok = (n > 0 and all(k in rows and rows[k].get("split") == fz["split"] for k in range(1, n + 1))
              and valid_features(t / "feats" / f"{q}.npz", (n, len(f["layers"]))))
        if ok:
            with np.load(t / "feats" / f"{q}.npz") as z:
                ok = z["mean"].shape[-1] == width
        if not ok:
            bad.append(q)
    if bad:
        problems.append(f"{rel(t)}: {len(bad)} episodes with missing steps, wrong split or unreadable features "
                        f"(e.g. {sorted(bad)[:3]})")
    return problems


def check(args, fz, pairs):
    problems = [f"code changed: {c}" for c, h in fz["code"].items() if sha256(ROOT / c) != h]
    if sha256(args.manifest) != fz["heldout_manifest"]["sha256"]:
        problems.append("held-out manifest changed")
    ids = json.loads(args.manifest.read_text(encoding="utf-8"))["splits"][fz["split"]]
    for s, t in pairs:
        f = fz["models"][s]
        changed = [k for k, h in f["files"].items() if sha256(ROOT / s / SOURCE_FILES[k]) != h]
        problems += [f"{s}: source file changed: {k}" for k in changed]
        with np.load(ROOT / s / "trivia_feats.npz") as z:
            width = z["mean"].shape[-1]
        problems += check_target(t, f, fz, ids, width)
    return problems


def score(fz, pairs, staging):
    # A resumed CPU analysis must see the same trajectories as its completed read-outs.
    target_hashes = {}
    for s, t in pairs:
        target = ROOT / t
        paths = [target / name for name in ("meta.json", "episodes.jsonl", "steps.jsonl")]
        paths += sorted((target / "feats").glob("*.npz"))
        target_hashes[s] = {p.relative_to(target).as_posix(): sha256(p) for p in paths}
    inputs = staging / "target_hashes.json"
    if inputs.exists():
        if json.loads(inputs.read_text(encoding="utf-8")) != target_hashes:
            raise SystemExit("Target inputs changed after confirmation scoring began")
    else:
        write_json(inputs, target_hashes)
    per_model, stage = {}, {}
    for k, (s, t) in enumerate(pairs):
        stage[s] = staging / f"model{k}"  # never derived from the run path
        for readout in ("upper", "selected"):
            completed = stage[s] / f".{readout}.complete"
            if completed.is_file():
                continue
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "20_source_swap.py"), "--run", str(ROOT / s),
                                "--target-run", str(ROOT / t), "--target-split", fz["split"], "--readout", readout,
                                "--results-dir", str(stage[s]), "--pca", str(fz["scoring"]["pca"]),
                                "--n-boot", str(fz["scoring"]["n_boot"]), "--min-m", str(fz["scoring"]["min_m"])],
                               capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"scripts/20 --readout {readout} failed on {t} (nothing written outside "
                                 f"{rel(staging)}):\n{r.stderr}")
            write_json(completed, True)
        per_model[s] = {ro: json.loads((stage[s] / f"{pre}tests.json").read_text(encoding="utf-8"))
                        for ro, pre in (("upper", "source_swap_upper_"), ("selected", "source_swap_"))}
    for s, t in pairs:  # retain staged copies until the final summary commits
        dest = ROOT / t / "results"
        dest.mkdir(exist_ok=True)
        for f in stage[s].iterdir():
            if not f.name.startswith("."):
                with f.open("rb") as src, atomic_open(dest / f.name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return per_model, sha256(inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", help="source_run=target_run per model (freeze only; default qwen7b, llama8b)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/heldout_v1/heldout_manifest.json")
    ap.add_argument("--split", default="final")
    ap.add_argument("--frozen", type=Path, default=ROOT / "manifests/source_swap/v1.2_frozen_inputs.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/source_swap_confirmation.json")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pca", type=int, help=argparse.SUPPRESS)  # synthetic freezes only
    ap.add_argument("--n-boot", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--min-m", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    with run_lock(args.frozen.parent):
        return run(args)


def run(args):
    supplied = any(v is not None for v in (args.pca, args.n_boot, args.min_m))
    if args.freeze:
        if args.frozen.resolve() == ROOT / "manifests/source_swap/v1.2_frozen_inputs.json":
            if args.pairs or supplied or args.split != "final":
                raise SystemExit("Use the registered models, final split and scoring settings for the official freeze")
        args.pca = 64 if args.pca is None else args.pca
        args.n_boot = 2000 if args.n_boot is None else args.n_boot
        args.min_m = 300 if args.min_m is None else args.min_m
        return freeze(args, [tuple(p.split("=", 1)) for p in args.pairs or DEFAULT_PAIRS])
    if args.pairs:
        raise SystemExit("--pairs is for --freeze only: the confirmation uses exactly the frozen model set")
    if supplied:
        raise SystemExit("Scoring parameters were fixed at freeze time")

    fz = json.loads(args.frozen.read_text(encoding="utf-8"))
    if args.out.resolve() != (ROOT / fz["output"]).resolve():
        raise SystemExit("Use the frozen output path for this confirmation")
    frozen_hash = sha256(args.frozen)
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("frozen_sha256") != frozen_hash:
            raise SystemExit("The output belongs to another freeze")
        print(f"Confirmation already completed: {rel(args.out)}")
        return
    pairs = [(s, f["target_run"]) for s, f in fz["models"].items()]
    problems = check(args, fz, pairs)
    if problems:
        raise SystemExit("not scoring; fix these first:\n  " + "\n  ".join(problems))
    staging = args.out.parent / f".source_swap_confirmation_{frozen_hash}"
    per_model, target_hash = score(fz, pairs, staging)

    flags = {s: {k: v for k, v in o["upper"]["tests"].items() if k.endswith("_this_model")}
             for s, o in per_model.items()}
    name, text = framing(flags)
    est = lambda tests: {k: tests[v] for k, v in KEY.items()}  # noqa: E731
    out = {"protocol": "source_swap_v1.2 (section 11)", "split": fz["split"], "n_boot": fz["scoring"]["n_boot"],
           "frozen_sha256": frozen_hash, "target_manifest_sha256": target_hash,
           "models": list(per_model),
           "H1_confirmed": all(f["H1_this_model"] for f in flags.values()),
           "gain_confirmed": all(f["gain_this_model"] for f in flags.values()),
           "framing": name, "framing_text": text, "flags": flags,
           "I_G_C": {s: est(o["upper"]["tests"]) for s, o in per_model.items()},
           "estimates_upper": {s: {k: v for k, v in o["upper"]["tests"].items() if isinstance(v, dict)}
                               for s, o in per_model.items()},
           "v1_1_selected_rule": {s: {"readouts": o["selected"]["readouts"], **est(o["selected"]["tests"])}
                                  for s, o in per_model.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out)
    shutil.rmtree(staging)
    for s in per_model:
        u, v = out["I_G_C"][s], out["v1_1_selected_rule"][s]
        print(s, "upper:", {k: round(x["estimate"], 3) for k, x in u.items()},
              "| v1.1 rule:", {k: round(v[k]["estimate"], 3) for k in KEY}, flags[s])
    print(f"H1'' confirmed: {out['H1_confirmed']}; gain confirmed: {out['gain_confirmed']}; framing: {name} ({text})")


if __name__ == "__main__":
    main()
