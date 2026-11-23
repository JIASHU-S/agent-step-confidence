"""Source-swap analysis, protocol v1.1 section 8 (protocols/source_swap_v1.md). CPU; run after 19 (main role).

    python scripts/20_source_swap.py --run runs/qwen7b

Sources: S0 (TriviaQA, QA format; the run's trivia.jsonl), C1 (TriviaQA, ReAct format; sources/trivia_react-main),
S2QA (2Wiki lookup choice, QA format; sources/wiki_search_qa-main). Each is balanced to m positives and m negatives
(m = smallest class count over the three), split 80/20 by hash, its read-out (pooling x layer) chosen on the 20%
by AUROC, and refit on all 2m items at that read-out. No agent label is read before scoring.

Targets: legacy test agent steps. Same-step AUROC (correct/wrong pairs with equal t) on finish, search ok vs
distractor (primary), search ok vs repeat and all steps (exploratory); pooled AUROC as well. CIs: 2000 question
bootstrap replicates (seed 0) shared by all sources.

H1' (primary): I = [A_dis(S2QA) - A_dis(S0)] - [A_fin(S2QA) - A_fin(S0)] > 0 (95% CI above 0; judged only once both
models are done). H2': A_dis(S2QA) - A_dis(S0) > 0 and A_fin(S0) - A_fin(S2QA) lower bound > -0.05. Format: C1 - S0.

Writes to <run>/results/: source_swap_readouts.csv, source_swap_auroc.csv, source_swap_layers.csv,
source_swap_tests.json

--readout upper (protocol v1.2 draft, section 11): every source is read at the same fixed set of read-outs, the
upper half of the stored layers x {mean, last}; each A is the mean of the per-read-out same-step AUROCs, and the
bootstrap averages within each replicate. Outputs carry the prefix source_swap_upper_ instead. --target-run and
--target-split point the scoring at another agent run of the same model (the confirmation targets); the source
probes always come from --run.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.errtypes import auroc_pairs, error_types  # noqa: E402
from agentconf.io import read_jsonl  # noqa: E402
from agentconf.metrics import auroc  # noqa: E402
from agentconf.probes import fit_predict  # noqa: E402
from agentconf.runs import load_features, load_run  # noqa: E402
from agentconf.sources import balance_indices, source_split  # noqa: E402

CONTRASTS = {"finish": ("finish_ok", "finish_wrong"), "distractor": ("search_ok", "search_distractor"),
             "repeat": ("search_ok", "search_repeat"), "all": None}
PRIMARY = ("finish", "distractor")
FINISH_MARGIN = 0.05  # v1.2: "finish loss below 0.05" needs the lower bound of A_fin(S2QA) - A_fin(S0) above -0.05


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_source(run, name):
    if name == "S0":
        rows, feats = read_jsonl(run / "trivia.jsonl"), np.load(run / "trivia_feats.npz")
    else:
        d = run / "sources" / {"C1": "trivia_react-main", "S2QA": "wiki_search_qa-main"}[name]
        if not json.loads((d / "meta.json").read_text(encoding="utf-8")).get("complete"):
            raise SystemExit(f"{d} is incomplete: finish scripts/19 first")
        rows, feats = read_jsonl(d / "items.jsonl"), np.load(d / "feats.npz")
    return [r["id"] for r in rows], np.array([r["correct"] for r in rows], int), {p: feats[p] for p in ("mean", "last")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-m", type=int, default=300)
    ap.add_argument("--readout", choices=("selected", "upper"), default="selected")
    ap.add_argument("--target-run", type=Path, help="agent run to score (default: --run)")
    ap.add_argument("--target-split", default="test")
    ap.add_argument("--results-dir", type=Path, help="where to write (default: <target run>/results)")
    args = ap.parse_args()
    target_run = args.target_run or args.run
    res = args.results_dir or target_run / "results"
    res.mkdir(parents=True, exist_ok=True)
    prefix = "source_swap_" if args.readout == "selected" else "source_swap_upper_"
    meta = load_run(args.run)[0]
    tmeta, _, steps = load_run(target_run)
    layers = meta["layers"]
    if tmeta["layers"] != layers or tmeta["model"] != meta["model"]:
        raise SystemExit("target run must use the same model and stored layers as the source run")
    upper = [(p, l) for p in ("mean", "last") for l in layers[len(layers) // 2:]]

    # ---- sources: balance, split, choose read-out on source labels only
    src = {n: load_source(args.run, n) for n in ("S0", "C1", "S2QA")}
    m = min(min(int(y.sum()), int((1 - y).sum())) for _, y, _ in src.values())
    if m < args.min_m:
        raise SystemExit(f"m = {m} < {args.min_m}: extend S2-QA per protocol section 8 before any target scoring")
    probes, ro_rows = {}, []
    for name, (ids, y, F) in src.items():
        idx = balance_indices(ids, y, m)
        bid, by = [ids[i] for i in idx], y[idx]
        tr = np.array(source_split(bid))
        best = None
        for pool in ("mean", "last"):
            for j, l in enumerate(layers):
                X = F[pool][idx, j, :]
                a = auroc(fit_predict(X[tr], by[tr], X[~tr], n_components=args.pca), by[~tr])
                ro_rows.append({"source": name, "pooling": pool, "layer": l, "source_val_auroc": a, "m": m,
                                "n_source": len(ids), "source_base_rate": float(y.mean())})
                if best is None or a > best[2]:
                    best = (pool, l, a)
        probes[name] = {"idx": idx, "y": by, "readout": best}
        print(f"{name}: n={len(ids)} base={y.mean():.3f} m={m} read-out {best[0]} L{best[1]} (source val AUROC {best[2]:.3f})")
    write_csv(res / f"{prefix}readouts.csv", ro_rows)

    # ---- targets
    te = np.array([s["split"] == args.target_split for s in steps])
    if not te.any():
        raise SystemExit(f"no steps with split {args.target_split!r} in {target_run}")
    yt = np.array([s["correct"] for s in steps], int)[te]
    tt = np.array([s["t"] for s in steps], int)[te]
    qt = np.array([s["qid"] for s in steps])[te]
    types = error_types(steps)[te]
    masks = {c: (np.ones(len(yt), bool) if mem is None else np.isin(types, mem)) for c, mem in CONTRASTS.items()}
    agent = {p: load_features(target_run, steps, p)[te] for p in ("mean", "last")}

    def score(name, pool, layer):
        _, _, F = src[name]
        j = layers.index(layer)
        p = probes[name]
        return fit_predict(F[pool][p["idx"], j, :], p["y"], agent[pool][:, j, :], n_components=args.pca)

    readouts = {n: [probes[n]["readout"][:2]] if args.readout == "selected" else upper for n in src}
    scores = {n: [score(n, *ro) for ro in readouts[n]] for n in src}

    def A(ss, c, same_step, idx):  # mean over the source's read-outs (one read-out under --readout selected)
        sub = idx[masks[c][idx]]
        return float(np.mean([auroc_pairs(s[sub], yt[sub], tt[sub] if same_step else None) for s in ss]))

    def derived(a):  # a[(source, contrast)] same-step AUROC
        d = {"I": (a["S2QA", "distractor"] - a["S0", "distractor"]) - (a["S2QA", "finish"] - a["S0", "finish"]),
             "H2_distractor_S2QA_minus_S0": a["S2QA", "distractor"] - a["S0", "distractor"],
             "H2_finish_S0_minus_S2QA": a["S0", "finish"] - a["S2QA", "finish"],
             "finish_S2QA_minus_S0": a["S2QA", "finish"] - a["S0", "finish"]}
        for c in CONTRASTS:
            d[f"format_C1_minus_S0_{c}"] = a["C1", c] - a["S0", c]
        return d

    full = np.arange(len(yt))
    point = {(n, c, v): A(scores[n], c, v == "same_step", full)
             for n in scores for c in CONTRASTS for v in ("same_step", "pooled")}
    uq = np.unique(qt)
    members = [np.flatnonzero(qt == u) for u in uq]
    rng = np.random.default_rng(0)
    reps, dreps = [], []
    for _ in range(args.n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uq), len(uq))])
        r = {(n, c, v): A(scores[n], c, v == "same_step", idx) for n, c, v in point}
        reps.append(r)
        dreps.append(derived({(n, c): r[(n, c, "same_step")] for n in scores for c in CONTRASTS}))

    rows = []
    for key, a in point.items():
        vals = np.array([r[key] for r in reps])
        rows.append({"source": key[0], "contrast": key[1], "version": key[2], "auroc": a,
                     "ci_lo": np.nanpercentile(vals, 2.5), "ci_hi": np.nanpercentile(vals, 97.5),
                     "n": int(masks[key[1]].sum()), "n_wrong": int((masks[key[1]] & (yt == 0)).sum())})
    write_csv(res / f"{prefix}auroc.csv", rows)

    dpoint = derived({(n, c): point[(n, c, "same_step")] for n in scores for c in CONTRASTS})
    tests = {}
    for k, v in dpoint.items():
        vals = np.array([d[k] for d in dreps])
        tests[k] = {"estimate": v, "ci_lo": float(np.nanpercentile(vals, 2.5)), "ci_hi": float(np.nanpercentile(vals, 97.5))}
    tests["H1_this_model"] = bool(tests["I"]["ci_lo"] > 0)
    if args.readout == "selected":  # v1.1 as frozen (its finish condition bounds S2QA's advantage, not its loss)
        tests["H2_this_model"] = bool(tests["H2_distractor_S2QA_minus_S0"]["ci_lo"] > 0
                                      and tests["H2_finish_S0_minus_S2QA"]["ci_lo"] > -0.05)
    else:  # v1.2: retrieval gain and finish cost reported separately (section 11.4)
        cost = tests["finish_S2QA_minus_S0"]
        tests["gain_this_model"] = bool(tests["H2_distractor_S2QA_minus_S0"]["ci_lo"] > 0)
        tests["finish_noninferior_this_model"] = bool(cost["ci_lo"] > -FINISH_MARGIN)
        tests["finish_cost_this_model"] = bool(cost["ci_hi"] < 0)
    out = {"protocol": "source_swap_v1.1 (section 8)" if args.readout == "selected" else "source_swap_v1.2 draft (section 11)",
           "model": meta["model"], "m": m, "readout_rule": args.readout,
           "target_run": str(target_run), "target_split": args.target_split,
           "readouts": {n: {"pooling": p["readout"][0], "layer": p["readout"][1], "source_val_auroc": p["readout"][2]}
                        for n, p in probes.items()} if args.readout == "selected"
           else {n: [list(ro) for ro in upper] for n in src},
           "tests": tests, "n_boot": args.n_boot,
           "note": "H1' is judged only when both models are done; this file holds one model."}
    (res / f"{prefix}tests.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in rows:
        if r["version"] == "same_step":
            print(f"  {r['source']:5s} {r['contrast']:10s} {r['auroc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
                  f"n={r['n']} wrong={r['n_wrong']}")
    for k, v in tests.items():
        print(f"  {k}: {v}")

    # ---- exploratory: every read-out of every source on the primary targets (descriptive)
    if args.readout == "upper":
        return
    layer_rows = []
    for n in src:
        for pool in ("mean", "last"):
            for l in layers:
                s = [score(n, pool, l)]
                layer_rows.append({"source": n, "pooling": pool, "layer": l,
                                   **{f"same_step_{c}": A(s, c, True, full) for c in CONTRASTS}})
    write_csv(res / f"{prefix}layers.csv", layer_rows)


if __name__ == "__main__":
    main()
