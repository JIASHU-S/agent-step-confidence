"""Robustness checks for RQ1 and RQ3 (CPU only; run after 03 and 04).

    python scripts/06_robustness.py --run runs/qwen7b

1. Episode-level (grouped) bootstrap for AUROC differences between the probe and each baseline. Steps of one
   trajectory are not independent, so whole episodes are resampled.
2. Lenient labels: a Search that returns a supporting paragraph counts as correct even if it was retrieved
   before. RQ1 is recomputed (probe refitted with the same pooling and layer as in 03).
3. RQ3 uncertainty: episode bootstrap for the hand-off curves (bands), the area under the risk-coverage curve of
   the hand-off policy, the trajectory ECE gap (product - forward) and the AUROC gap (forward - last step).

Writes to <run>/results/: robust_rq1_bootstrap.csv, robust_lenient_rq1.csv, robust_rq3.json,
robust_handoff_bands.csv, figures/handoff_bands.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.metrics import aurc, auroc, ece, platt, summarize  # noqa: E402
from agentconf.probes import fit_predict, oof_scores  # noqa: E402
from agentconf.runs import load_features, load_run  # noqa: E402

BASELINES = ["consistency", "verbalized", "lp_mean", "lp_min"]


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def grouped_bootstrap(groups, n_boot, seed, stat):
    """stat(index_array) -> float, evaluated on episode-resampled step indices."""
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ids, len(ids))])
        out.append(stat(idx))
    return out


def lenient_labels(steps):
    """Search is correct if its paragraph is a supporting one, repeats included. Supporting titles are exactly the
    titles whose first retrieval was labelled correct under the strict rule."""
    gold, out = {}, []
    for s in steps:
        seen = gold.setdefault(s["qid"], set())
        if s["kind"] == "Search" and s["correct"]:
            seen.add(s["title"])
        lenient = int(s["correct"] or (s["kind"] == "Search" and s["title"] in seen))
        out.append(lenient)
    return np.array(out)


def handoff_stats(min_conf, success, taus):
    rows = []
    for tau in taus:
        kept = min_conf >= tau
        rows.append((kept.mean(), success[kept].mean() if kept.any() else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()
    res = args.run / "results"

    # ---- 1. grouped bootstrap on the test steps
    scores = read_csv(res / "step_scores.csv")
    te = [r for r in scores if r["split"] == "test"]
    g = np.array([r["qid"] for r in te])
    y = np.array([int(r["correct"]) for r in te])
    sig = {k: np.nan_to_num(np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in te]))
           for k in ["probe", *BASELINES]}
    rows = []
    for b in BASELINES:
        d = grouped_bootstrap(g, args.n_boot, 0, lambda i: auroc(sig["probe"][i], y[i]) - auroc(sig[b][i], y[i]))
        m, lo, hi = ci(d)
        rows.append({"comparison": f"probe - {b}", "delta_auroc": m, "ci_low": lo, "ci_high": hi,
                     "resampling": "episodes", "n_episodes": len(np.unique(g))})
    write_csv(res / "robust_rq1_bootstrap.csv", rows)
    print("grouped bootstrap:", json.dumps(rows, indent=1))

    # ---- 2. lenient labels
    _, episodes, steps = load_run(args.run)
    yl = lenient_labels(steps)
    tr = np.array([s["split"] == "train" for s in steps])
    groups = np.array([s["qid"] for s in steps])
    choice = json.loads((res / "probe_choice.json").read_text(encoding="utf-8"))
    meta_layers = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))["layers"]
    X = load_features(args.run, steps, choice["pooling"])[:, meta_layers.index(choice["layer"]), :]
    probe = np.full(len(steps), np.nan)
    probe[tr] = oof_scores(X[tr], yl[tr], groups[tr], n_components=args.pca)
    probe[~tr] = fit_predict(X[tr], yl[tr], X[~tr], n_components=args.pca)
    changed = int((yl != np.array([s["correct"] for s in steps])).sum())
    out = []
    for name in ["probe", *BASELINES]:
        sc = probe if name == "probe" else np.array(
            [s[name] if s[name] is not None else np.nan for s in steps], float)
        sc = np.nan_to_num(sc, nan=np.nanmedian(sc[tr]))
        prob = sc[~tr] if name == "probe" else platt(sc[tr], yl[tr], sc[~tr])
        out.append({"signal": name, "labels": "lenient", **summarize(sc[~tr], prob, yl[~tr]),
                    "labels_changed_all_splits": changed})
    write_csv(res / "robust_lenient_rq1.csv", out)
    print(f"lenient labels changed {changed} of {len(steps)} steps")
    print(json.dumps(out, indent=1))

    # ---- 3. RQ3 uncertainty
    ep = read_csv(res / "rq3_episodes.csv")
    succ = np.array([int(r["success"]) for r in ep])
    m0 = np.array([min(json.loads(r["running_M0"])) for r in ep])
    m1 = np.array([min(json.loads(r["running_M1"])) for r in ep])
    prod = np.array([float(r["M0_product"]) for r in ep])
    fwd = np.array([float(r["M1_forward"]) for r in ep])
    last = np.array([float(r["M0_last"]) for r in ep])
    rng = np.random.default_rng(1)
    taus = np.linspace(0.05, 0.95, 19)
    boots = {"aurc_M0_minus_M1": [], "ece_product_minus_forward": [], "auroc_forward_minus_last": []}
    curves = {"M0": [], "M1": []}
    n = len(succ)
    for _ in range(args.n_boot):
        i = rng.integers(0, n, n)
        boots["aurc_M0_minus_M1"].append(aurc(m0[i], succ[i]) - aurc(m1[i], succ[i]))
        boots["ece_product_minus_forward"].append(ece(prod[i], succ[i]) - ece(fwd[i], succ[i]))
        boots["auroc_forward_minus_last"].append(auroc(fwd[i], succ[i]) - auroc(last[i], succ[i]))
        curves["M0"].append(handoff_stats(m0[i], succ[i], taus))
        curves["M1"].append(handoff_stats(m1[i], succ[i], taus))
    summary = {"n_episodes": n, "aurc_M0": aurc(m0, succ), "aurc_M1": aurc(m1, succ)}
    for k, v in boots.items():
        m, lo, hi = ci(v)
        summary[k] = {"mean": m, "ci_low": lo, "ci_high": hi}
    (res / "robust_rq3.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))

    band_rows = []
    for model, mc in [("M0", m0), ("M1", m1)]:
        point = handoff_stats(mc, succ, taus)
        arr = np.array(curves[model], float)  # [B, tau, 2]
        for j, tau in enumerate(taus):
            acc = arr[:, j, 1]
            acc = acc[np.isfinite(acc)]
            band_rows.append({"model": model, "tau": float(tau), "coverage": point[j][0],
                              "kept_accuracy": point[j][1],
                              "acc_low": float(np.percentile(acc, 2.5)) if len(acc) > 50 else np.nan,
                              "acc_high": float(np.percentile(acc, 97.5)) if len(acc) > 50 else np.nan})
    write_csv(res / "robust_handoff_bands.csv", band_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for model, label in [("M0", "independent steps (product)"), ("M1", "coupled errors (forward)")]:
        r = [b for b in band_rows if b["model"] == model and np.isfinite(b["acc_low"])]
        cov = [b["coverage"] for b in r]
        line, = ax.plot(cov, [b["kept_accuracy"] for b in r], marker="o", ms=3, label=label)
        ax.fill_between(cov, [b["acc_low"] for b in r], [b["acc_high"] for b in r], alpha=0.2,
                        color=line.get_color(), lw=0)
    ax.set_xlabel("share of episodes kept by the agent")
    ax.set_ylabel("accuracy of kept episodes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    (res / "figures").mkdir(exist_ok=True)
    fig.savefig(res / "figures" / "handoff_bands.pdf")
    fig.savefig(res / "figures" / "handoff_bands.png", dpi=200)


if __name__ == "__main__":
    main()
