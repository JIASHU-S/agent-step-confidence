"""How early can a wrong step be detected? Probes on pre_step / pre_action states vs the Action-span probe (CPU).

    python scripts/08_preaction_analysis.py --run runs/qwen7b

Layer for each read-out point is chosen by out-of-fold AUROC on the training split only. Writes to
<run>/results/: preaction_layers.csv, preaction.csv (test AUROC/Brier/ECE per read-out point, with an
episode-bootstrap CI of the gap to the Action-span probe and to sampling consistency), figures/preaction_layers.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.metrics import auroc, summarize  # noqa: E402
from agentconf.probes import fit_predict, layer_sweep  # noqa: E402
from agentconf.runs import load_run  # noqa: E402

POINTS = ["pre_step", "pre_action"]


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    res = args.run / "results"
    layers = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))["layers"]

    _, _, steps = load_run(args.run)
    scores = {(r["qid"], int(r["t"])): r for r in read_csv(res / "step_scores.csv")}
    cache, feats = {}, {p: [] for p in POINTS}
    for s in steps:
        if s["qid"] not in cache:
            cache[s["qid"]] = np.load(args.run / "feats_pre" / f"{s['qid']}.npz")
        for p in POINTS:
            feats[p].append(cache[s["qid"]][p][s["t"] - 1])
    feats = {p: np.stack(v).astype(np.float32) for p, v in feats.items()}
    y = np.array([s["correct"] for s in steps])
    tr = np.array([s["split"] == "train" for s in steps])
    te = ~tr
    groups = np.array([s["qid"] for s in steps])
    action_probe = np.array([float(scores[(s["qid"], s["t"])]["probe"]) for s in steps])
    consistency = np.array([float(scores[(s["qid"], s["t"])]["consistency"]) for s in steps])

    layer_rows, preds = [], {}
    for p in POINTS:
        sweep = layer_sweep(feats[p][tr], y[tr], groups[tr], layers, n_components=args.pca)
        for j, l in enumerate(layers):
            test = auroc(fit_predict(feats[p][tr, j], y[tr], feats[p][te, j], n_components=args.pca), y[te])
            layer_rows.append({"point": p, "layer": l, "oof_auroc_train": sweep[l], "test_auroc": test})
        best = max(sweep, key=lambda l: np.nan_to_num(sweep[l], nan=-1))
        j = layers.index(best)
        preds[p] = (best, fit_predict(feats[p][tr, j], y[tr], feats[p][te, j], n_components=args.pca))
    write_csv(res / "preaction_layers.csv", layer_rows)

    gte = groups[te]
    ids = np.unique(gte)
    members = {g: np.flatnonzero(gte == g) for g in ids}
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([members[g] for g in rng.choice(ids, len(ids))]) for _ in range(args.n_boot)]

    def gap(a, b):
        d = [auroc(a[i], y[te][i]) - auroc(b[i], y[te][i]) for i in boot_idx]
        return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    rows = [{"point": "action (03 probe)", "layer": json.loads((res / "probe_choice.json").read_text())["layer"],
             **summarize(action_probe[te], action_probe[te], y[te])}]
    for p in POINTS:
        layer, pr = preds[p]
        g1, g2 = gap(pr, action_probe[te]), gap(pr, consistency[te])
        rows.append({"point": p, "layer": layer, **summarize(pr, pr, y[te]),
                     "minus_action_probe": g1[0], "minus_action_probe_ci": f"[{g1[1]:.3f}, {g1[2]:.3f}]",
                     "minus_consistency": g2[0], "minus_consistency_ci": f"[{g2[1]:.3f}, {g2[2]:.3f}]"})
    write_csv(res / "preaction.csv", rows)
    print(json.dumps(rows, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    action_layers = read_csv(res / "rq1_layers.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    pooling = json.loads((res / "probe_choice.json").read_text())["pooling"]
    r = [x for x in action_layers if x["pooling"] == pooling]
    ax.plot([int(x["layer"]) for x in r], [float(x["test_auroc"]) for x in r], marker="o",
            label=f"Action tokens ({pooling})")
    for p, label in [("pre_action", "after Thought, before Action"), ("pre_step", "before the step")]:
        r = [x for x in layer_rows if x["point"] == p]
        ax.plot([x["layer"] for x in r], [x["test_auroc"] for x in r], marker="s", label=label)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("step-level AUROC (test)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    (res / "figures").mkdir(exist_ok=True)
    fig.savefig(res / "figures" / "preaction_layers.pdf")
    fig.savefig(res / "figures" / "preaction_layers.png", dpi=200)


if __name__ == "__main__":
    main()
