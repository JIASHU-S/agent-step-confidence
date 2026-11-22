"""Does the probe rank steps within one trajectory, or mostly separate easy questions from hard ones? (CPU; run after 03.)

    python scripts/15_within_question.py --run runs/qwen7b

Pooled AUROC mostly compares steps from different questions, so it can partly reflect question difficulty.
Here AUROC is computed only over pairs of one correct and one wrong test step from the SAME question (each
question has one trajectory). Pairs are pooled over questions, not averaged per question. Signals: the probe,
consistency, a step-index baseline (score = -t), the probe logit residualised on [1, t, is_Finish] by least
squares on the training split, and the probe on Search steps only. CIs resample test questions (2000 replicates,
seed 0); the probe minus step-index gap is computed on the same replicates.

Writes to <run>/results/: within_question.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pair_counts(score, y, members):
    """Per question: (concordant + ties/2, number of pairs) over its correct x wrong step pairs."""
    win, n = np.zeros(len(members)), np.zeros(len(members))
    for j, idx in enumerate(members):
        pos, neg = score[idx][y[idx] == 1], score[idx][y[idx] == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            win[j] = (d > 0).sum() + 0.5 * (d == 0).sum()
            n[j] = d.size
    return win, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    res = args.run / "results"
    rows = read_csv(res / "step_scores.csv")

    split = np.array([r["split"] for r in rows])
    qid = np.array([r["qid"] for r in rows])
    t = np.array([int(r["t"]) for r in rows], float)
    is_finish = np.array([r["kind"] == "Finish" for r in rows], float)
    y = np.array([int(r["correct"]) for r in rows])
    probe = logit(np.array([float(r["probe"]) for r in rows]))
    consistency = np.array([float(r["consistency"]) for r in rows])

    # Residualise the probe logit on position and action type; coefficients from the training split only.
    X = np.column_stack([np.ones(len(rows)), t, is_finish])
    tr = split == "train"
    beta = np.linalg.lstsq(X[tr], probe[tr], rcond=None)[0]
    resid = probe - X @ beta

    te = split == "test"
    search = te & (is_finish == 0)
    ids = np.unique(qid[te])
    sig = {"probe": (probe, te), "consistency": (consistency, te), "step index": (-t, te),
           "probe residualised": (resid, te), "probe, Search only": (probe, search)}
    counts = {k: pair_counts(s, y, [np.flatnonzero(m & (qid == g)) for g in ids]) for k, (s, m) in sig.items()}

    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(ids), len(ids)) for _ in range(args.n_boot)]

    def boot_auroc(k):
        win, n = counts[k]
        return np.array([win[b].sum() / n[b].sum() for b in boot])

    out = []
    for k in sig:
        win, n = counts[k]
        a = boot_auroc(k)
        out.append({"signal": k, "auroc": win.sum() / n.sum(), "ci_low": np.percentile(a, 2.5),
                    "ci_high": np.percentile(a, 97.5), "n_pairs": int(n.sum()),
                    "n_questions_with_pairs": int((n > 0).sum()), "n_test_questions": len(ids)})
    d = boot_auroc("probe") - boot_auroc("step index")
    point = out[0]["auroc"] - out[2]["auroc"]
    out.append({"signal": "probe - step index", "auroc": point, "ci_low": np.percentile(d, 2.5),
                "ci_high": np.percentile(d, 97.5), "n_pairs": out[0]["n_pairs"],
                "n_questions_with_pairs": out[0]["n_questions_with_pairs"], "n_test_questions": len(ids)})
    out = [{k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()} for r in out]
    write_csv(res / "within_question.csv", out)
    print(json.dumps({"residual_beta": beta.tolist(), "rows": out}, indent=1))


if __name__ == "__main__":
    main()
