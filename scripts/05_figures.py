"""Figures for the report from <run>/results/*.csv.

    python scripts/05_figures.py --run runs/qwen7b
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    res = args.run / "results"
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Probe AUROC by layer (in-domain test, and zero-shot transfer from TriviaQA if available).
    layers = rows(res / "rq1_layers.csv")
    transfer = rows(res / "rq2_layers.csv") if (res / "rq2_layers.csv").exists() else []
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for pooling, style in [("mean", "-"), ("last", "--")]:
        r = [x for x in layers if x["pooling"] == pooling]
        ax.plot([int(x["layer"]) for x in r], [float(x["test_auroc"]) for x in r], style, marker="o",
                label=f"agent-trained ({pooling})")
        t = [x for x in transfer if x["pooling"] == pooling]
        if t:
            ax.plot([int(x["layer"]) for x in t], [float(x["zero_shot_auroc"]) for x in t], style, marker="x",
                    label=f"TriviaQA-trained ({pooling})")
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("step-level AUROC (test)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "probe_layers.pdf")
    fig.savefig(fig_dir / "probe_layers.png", dpi=200)

    # Hand-off curves: share of episodes the agent keeps vs accuracy on them.
    if (res / "rq3_handoff.csv").exists():
        h = rows(res / "rq3_handoff.csv")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for model, label in [("M0", "independent steps (product)"), ("M1", "coupled errors (forward)")]:
            r = [x for x in h if x["model"] == model and x["kept_accuracy"] not in ("", "nan")]
            ax.plot([float(x["coverage"]) for x in r], [float(x["kept_accuracy"]) for x in r], marker="o",
                    label=label)
        ax.set_xlabel("share of episodes kept by the agent")
        ax.set_ylabel("accuracy of kept episodes")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "handoff.pdf")
        fig.savefig(fig_dir / "handoff.png", dpi=200)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
