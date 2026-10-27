"""RQ1 (which signal detects a wrong step?) and RQ2 (do probes trained on single-step QA transfer to agent steps?).

    python scripts/03_step_signals.py --run runs/qwen7b

Writes to <run>/results/:
    rq1_layers.csv     out-of-fold AUROC per layer and pooling on the training split (used to pick the layer)
    rq1_signals.csv    test-split AUROC / Brier / ECE per signal, overall and per action kind
    rq1_bootstrap.csv  paired bootstrap CI of AUROC(probe) - AUROC(each baseline)
    step_scores.csv    per-step scores: out-of-fold on train, held-out on test (input to RQ3)
    rq2_transfer.csv   TriviaQA-trained probe on agent test steps: zero-shot, recalibrated, vs in-domain few-shot
    rq2_layers.csv     zero-shot transfer AUROC per layer
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.io import read_jsonl  # noqa: E402
from agentconf.metrics import auroc, paired_bootstrap, platt, summarize  # noqa: E402
from agentconf.probes import fit_predict, layer_sweep, oof_scores  # noqa: E402
from agentconf.runs import load_features, load_run  # noqa: E402

BASELINES = ["lp_mean", "lp_min", "consistency", "verbalized"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--few-shot", type=int, default=50, help="agent steps used to recalibrate / train in RQ2")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    trivia_meta = args.run / "trivia_meta.json"
    if trivia_meta.exists() and not json.loads(trivia_meta.read_text(encoding="utf-8")).get("complete", False):
        raise ValueError("TriviaQA collection/export is incomplete; resume scripts/02_trivia_features.py first")
    res = args.run / "results"

    meta, episodes, steps = load_run(args.run)
    layers = meta["layers"]
    tr = np.array([s["split"] == "train" for s in steps])
    te = ~tr
    y = np.array([s["correct"] for s in steps], int)
    groups = np.array([s["qid"] for s in steps])
    kinds = np.array([s["kind"] for s in steps])
    print(f"{len(steps)} steps ({tr.sum()} train / {te.sum()} test), step accuracy {y.mean():.3f}, "
          f"episode success {np.mean([e['success'] for e in episodes]):.3f}")

    # ---- RQ1: choose pooling + layer on the training split only.
    feats, sweep_rows = {}, []
    for pooling in ("mean", "last"):
        feats[pooling] = load_features(args.run, steps, pooling)
        sweep = layer_sweep(feats[pooling][tr], y[tr], groups[tr], layers, n_components=args.pca)
        sweep_rows += [{"pooling": pooling, "layer": l, "oof_auroc_train": a} for l, a in sweep.items()]
    best = max(sweep_rows, key=lambda r: np.nan_to_num(r["oof_auroc_train"], nan=-1))
    j = layers.index(best["layer"])
    X = feats[best["pooling"]][:, j, :]
    print(f"probe: pooling={best['pooling']} layer={best['layer']} (train OOF AUROC {best['oof_auroc_train']:.3f})")

    probe = np.full(len(steps), np.nan)
    probe[tr] = oof_scores(X[tr], y[tr], groups[tr], n_components=args.pca)
    probe[te] = fit_predict(X[tr], y[tr], X[te], n_components=args.pca)
    # Test-split AUROC for every layer as well, for the layer plot (selection above never saw these).
    for r in sweep_rows:
        jj = layers.index(r["layer"])
        F = feats[r["pooling"]][:, jj, :]
        r["test_auroc"] = auroc(fit_predict(F[tr], y[tr], F[te], n_components=args.pca), y[te])
    write_csv(res / "rq1_layers.csv", sweep_rows)

    signals = {"probe": probe}
    for b in BASELINES:
        signals[b] = np.array([s[b] if s[b] is not None else np.nan for s in steps], float)

    rows = []
    for name, sc in signals.items():
        prob = sc[te] if name == "probe" else platt(sc[tr], y[tr], sc[te])
        for kind in ["all", "Search", "Finish"]:
            m = np.ones(te.sum(), bool) if kind == "all" else kinds[te] == kind
            if m.sum() < 10:
                continue
            raw = np.nan_to_num(sc[te][m], nan=np.nanmedian(sc[tr]) if np.isfinite(sc[tr]).any() else 0.0)
            rows.append({"signal": name, "kind": kind, **summarize(raw, prob[m], y[te][m]),
                         "missing": int(np.isnan(sc[te][m]).sum())})
    write_csv(res / "rq1_signals.csv", rows)

    boot = []
    for b in BASELINES:
        sb = np.nan_to_num(signals[b][te], nan=0.0)
        mean, lo, hi = paired_bootstrap(auroc, probe[te], sb, y[te])
        boot.append({"comparison": f"probe - {b}", "delta_auroc": mean, "ci_low": lo, "ci_high": hi})
    write_csv(res / "rq1_bootstrap.csv", boot)

    score_rows = []
    for i, s in enumerate(steps):
        score_rows.append({"qid": s["qid"], "t": s["t"], "split": s["split"], "kind": s["kind"],
                           "correct": s["correct"], **{k: v[i] for k, v in signals.items()}})
    write_csv(res / "step_scores.csv", score_rows)

    # ---- RQ2: transfer from TriviaQA.
    tv_path = args.run / "trivia_feats.npz"
    if not tv_path.exists():
        print("no trivia_feats.npz: skipping RQ2 (run scripts/02_trivia_features.py)")
        return
    tv = np.load(tv_path)
    ty = np.array([r["correct"] for r in read_jsonl(args.run / "trivia.jsonl")], int)
    print(f"TriviaQA: {len(ty)} answers, accuracy {ty.mean():.3f}")

    layer_rows = []
    for pooling in ("mean", "last"):
        for jj, l in enumerate(layers):
            p = fit_predict(tv[pooling][:, jj, :], ty, feats[pooling][te][:, jj, :], n_components=args.pca)
            layer_rows.append({"pooling": pooling, "layer": l, "zero_shot_auroc": auroc(p, y[te])})
    write_csv(res / "rq2_layers.csv", layer_rows)

    Xs = tv[best["pooling"]][:, j, :]
    transfer_te = fit_predict(Xs, ty, X[te], n_components=args.pca)
    transfer_tr = fit_predict(Xs, ty, X[tr], n_components=args.pca)
    out = [{"setting": "in-domain probe (all train steps)", **summarize(probe[te], probe[te], y[te])},
           {"setting": "TriviaQA probe, zero-shot", **summarize(transfer_te, transfer_te, y[te])}]
    rng = np.random.default_rng(0)
    tr_idx = np.flatnonzero(tr)
    recal, small = [], []
    for _ in range(args.reps):
        pick = rng.choice(tr_idx, size=min(args.few_shot, len(tr_idx)), replace=False)
        if len(set(y[pick])) < 2:
            continue
        pos = np.searchsorted(tr_idx, pick)
        recal.append(summarize(transfer_te, platt(transfer_tr[pos], y[pick], transfer_te), y[te]))
        p_small = fit_predict(X[pick], y[pick], X[te], n_components=min(args.pca, len(pick) - 1))
        small.append(summarize(p_small, p_small, y[te]))
    for name, runs in [(f"TriviaQA probe + Platt on {args.few_shot} agent steps", recal),
                       (f"in-domain probe on {args.few_shot} agent steps", small)]:
        if runs:
            row = {"setting": name}
            for k in ("auroc", "brier", "ece"):
                v = np.array([r[k] for r in runs])
                row[k], row[f"{k}_sd"] = float(np.nanmean(v)), float(np.nanstd(v))
            row["n"], row["reps"] = runs[0]["n"], len(runs)
            out.append(row)
    write_csv(res / "rq2_transfer.csv", out)
    (res / "probe_choice.json").write_text(json.dumps(best, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
