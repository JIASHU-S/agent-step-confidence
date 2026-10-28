"""Does the probe only read where the agent is in the episode? Position baselines and controls (CPU; run after 08).

    python scripts/09_position_control.py --run runs/qwen7b

Later Searches are more often repeats, and repeats are labelled wrong, so step position alone predicts errors.
1. Position baselines, fitted on the training split: step index only (known before the step, the fair comparison
   for the pre-step probe) and step index + action type (known once the action is written). Each is compared
   with the probes, and each probe is added to its baseline to measure what it contributes beyond position.
   CIs resample whole test episodes.
2. Probe AUROC within each step index, for all steps and for Search steps only.
3. RQ3: episode-bootstrap CI of the trajectory ECE gap between the last step (M0) and the forward recursion (M1).

Writes to <run>/results/: position_baseline.csv, position_strata.csv, position_rq3.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.metrics import auroc, ece, summarize  # noqa: E402
from agentconf.probes import fit_predict, oof_scores  # noqa: E402
from agentconf.runs import load_run  # noqa: E402


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def one_hot(values, levels):
    return np.array([[v == level for level in levels] for v in values], float)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_lr(X, y, tr):
    """Near-unregularised logistic regression fitted on the training split; returns probabilities for every row."""
    return LogisticRegression(C=1e4, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X)[:, 1]


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
    y = np.array([s["correct"] for s in steps])
    tr = np.array([s["split"] == "train" for s in steps])
    te = ~tr
    groups = np.array([s["qid"] for s in steps])
    t = np.array([s["t"] for s in steps])
    kind = np.array([s["kind"] for s in steps])
    # Train-split probe scores in step_scores.csv are out-of-fold, so they can be stacked with position safely.
    action_probe = np.array([float(scores[(s["qid"], s["t"])]["probe"]) for s in steps])
    consistency = np.array([float(scores[(s["qid"], s["t"])]["consistency"]) for s in steps])

    # Pre-step probe at the layer chosen by 08: out-of-fold on train, fitted on all of train for test.
    pre_layer = int(next(r for r in read_csv(res / "preaction.csv") if r["point"] == "pre_step")["layer"])
    cache, rows = {}, []
    for s in steps:
        if s["qid"] not in cache:
            cache[s["qid"]] = np.load(args.run / "feats_pre" / f"{s['qid']}.npz")["pre_step"]
        rows.append(cache[s["qid"]][s["t"] - 1, layers.index(pre_layer)])
    X = np.stack(rows).astype(np.float32)
    pre_step = np.full(len(steps), np.nan)
    pre_step[tr] = oof_scores(X[tr], y[tr], groups[tr], n_components=args.pca)
    pre_step[te] = fit_predict(X[tr], y[tr], X[te], n_components=args.pca)

    T = one_hot(t, sorted(set(t)))
    TK = np.hstack([T, one_hot(kind, sorted(set(kind)))])
    sig = {
        "step index": fit_lr(T, y, tr),
        "step index + action type": fit_lr(TK, y, tr),
        "probe: before step": pre_step,
        "probe: action tokens": action_probe,
        "step index + probe before step": fit_lr(np.hstack([T, logit(pre_step)[:, None]]), y, tr),
        "step index + action type + probe": fit_lr(np.hstack([TK, logit(action_probe)[:, None]]), y, tr),
        "consistency": consistency,
    }

    gte, yte = groups[te], y[te]
    ids = np.unique(gte)
    members = {g: np.flatnonzero(gte == g) for g in ids}
    rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([members[g] for g in rng.choice(ids, len(ids))]) for _ in range(args.n_boot)]

    def gap(a, b):
        a, b = sig[a][te], sig[b][te]
        d = [auroc(a[i], yte[i]) - auroc(b[i], yte[i]) for i in boot_idx]
        # delta_auroc is the bootstrap mean, as in 06 and 08; delta_auroc_point is the full-sample difference.
        return {"delta_auroc": float(np.mean(d)), "delta_auroc_point": auroc(a, yte) - auroc(b, yte),
                "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5))}

    comparisons = [("step index", "probe: action tokens"),
                   ("step index + action type", "probe: action tokens"),
                   ("probe: before step", "step index"),
                   ("step index + probe before step", "step index"),
                   ("probe: action tokens", "step index + action type"),
                   ("step index + action type + probe", "step index + action type"),
                   ("step index + action type + probe", "probe: action tokens")]
    out = [{"signal": k, **summarize(v[te], v[te], yte)} for k, v in sig.items()]
    out += [{"signal": f"{a} - {b}", **gap(a, b)} for a, b in comparisons]
    write_csv(res / "position_baseline.csv", out)
    print(json.dumps(out, indent=1))

    strata = []
    for subset, mask in [("all", te), ("Search", te & (kind == "Search"))]:
        for label, sel in [("1", t == 1), ("2", t == 2), ("3", t == 3), (">=4", t >= 4)]:
            m = mask & sel
            strata.append({"steps": subset, "t": label, "n": int(m.sum()), "base_rate": float(y[m].mean()),
                           **{f"auroc_{k}": auroc(sig[k][m], y[m])
                              for k in ["probe: action tokens", "probe: before step", "consistency"]}})
    write_csv(res / "position_strata.csv", strata)
    print(json.dumps(strata, indent=1))

    ep = read_csv(res / "rq3_episodes.csv")
    succ = np.array([int(r["success"]) for r in ep])
    last = np.array([float(r["M0_last"]) for r in ep])
    fwd = np.array([float(r["M1_forward"]) for r in ep])
    rng = np.random.default_rng(1)
    d = []
    for _ in range(args.n_boot):
        i = rng.integers(0, len(succ), len(succ))
        d.append(ece(last[i], succ[i]) - ece(fwd[i], succ[i]))
    summary = {"n_episodes": len(succ), "ece_last": ece(last, succ), "ece_forward": ece(fwd, succ),
               "ece_last_minus_forward": {"point": ece(last, succ) - ece(fwd, succ), "mean": float(np.mean(d)),
                                          "ci_low": float(np.percentile(d, 2.5)),
                                          "ci_high": float(np.percentile(d, 97.5))}}
    (res / "position_rq3.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
