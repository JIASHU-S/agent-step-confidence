"""RQ3: are step errors coupled, and does modelling that give better trajectory confidence and hand-off decisions?

    python scripts/04_trajectory.py --run runs/qwen7b --signal probe

Reads <run>/results/step_scores.csv (from 03). Train-split scores are out-of-fold, so the Bayesian models are fitted
on honest signals. Writes:
    rq3_loo.json          PSIS-LOO comparison of M0 (independent) vs M1 (coupled): DeltaELPD, SE, SR, Pareto k
    rq3_params.csv        posterior summaries
    rq3_trajectory.csv    test-split trajectory confidence metrics per aggregation method
    rq3_episodes.csv      per-episode confidences
    rq3_handoff.csv       hand-off curves (first step with running confidence < tau)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.metrics import summarize  # noqa: E402
from agentconf.runs import load_run  # noqa: E402
from agentconf.seqmodel import (compare, design, fit, handoff_curve, logit, posterior,  # noqa: E402
                                posterior_dataset, sigmoid, trajectory_confidences)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--signal", default="probe")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()
    res = args.run / "results"

    _, episodes, _ = load_run(args.run)
    ep_by_id = {e["qid"]: e for e in episodes}
    rows = read_csv(res / "step_scores.csv")
    vals = np.array([float(r[args.signal]) if r[args.signal] not in ("", "nan") else np.nan for r in rows])
    if args.signal != "probe":  # put a raw signal on a probability scale using the training split
        from agentconf.metrics import platt

        tr = np.array([r["split"] == "train" for r in rows])
        y = np.array([int(r["correct"]) for r in rows])
        vals = platt(vals[tr], y[tr], vals)
    steps = [{"qid": r["qid"], "t": int(r["t"]), "kind": r["kind"], "correct": int(r["correct"]),
              "split": r["split"], "s": float(logit(v))} for r, v in zip(rows, vals)]

    train = design([s for s in steps if s["split"] == "train"])
    id0 = fit(train, coupled=False, draws=args.draws, chains=args.chains)
    id1 = fit(train, coupled=True, draws=args.draws, chains=args.chains, seed=1)
    loo = compare(id0, id1)
    params = []
    for name, idata in [("M0", id0), ("M1", id1)]:
        post = posterior_dataset(idata)
        for var in post.data_vars:
            v = post[var].values.reshape(-1, *post[var].shape[2:])
            flat = v.reshape(v.shape[0], -1)
            for k in range(flat.shape[1]):
                params.append({"model": name, "param": f"{var}[{k}]" if flat.shape[1] > 1 else var,
                               "mean": float(flat[:, k].mean()), "q05": float(np.quantile(flat[:, k], 0.05)),
                               "q95": float(np.quantile(flat[:, k], 0.95))})
    write_csv(res / "rq3_params.csv", params)
    print(json.dumps(loo, indent=1))

    p0, p1 = posterior(id0), posterior(id1)
    by_ep = defaultdict(list)
    for s in steps:
        if s["split"] == "test":
            by_ep[s["qid"]].append(s)
    ep_rows, unfinished = [], 0
    for qid, st in by_ep.items():
        st.sort(key=lambda s: s["t"])
        ep = ep_by_id[qid]
        if not ep["finished"]:
            unfinished += 1  # the system knows these failed to answer; they are handed off regardless
            continue
        c = trajectory_confidences(st, p0, p1)
        a0, b0, _ = p0
        running0 = []  # M0 running product, for the hand-off comparison
        prod = np.ones(a0.shape[0])
        for s in st:
            k = ["Search", "Finish", "invalid"].index(s["kind"])
            prod = prod * sigmoid(a0[:, k] + b0[:, k] * s["s"])
            running0.append(float(prod.mean()))
        ep_rows.append({"qid": qid, "success": ep["success"], "n_steps": len(st),
                        **{k: v for k, v in c.items() if k != "running"},
                        "running_M1": json.dumps(c["running"]), "running_M0": json.dumps(running0)})
    write_csv(res / "rq3_episodes.csv", ep_rows)

    succ = np.array([r["success"] for r in ep_rows])
    metrics = []
    for m in ["M0_last", "M0_product", "M0_min", "M1_forward"]:
        conf = np.array([r[m] for r in ep_rows])
        metrics.append({"method": m, **summarize(conf, conf, succ)})
    write_csv(res / "rq3_trajectory.csv", metrics)
    print(f"{len(ep_rows)} finished test episodes ({unfinished} unfinished excluded)")
    print(json.dumps(metrics, indent=1))

    taus = np.linspace(0.05, 0.95, 19)
    curves = []
    for name in ["M0", "M1"]:
        eps = [{"running": json.loads(r[f"running_{name}"]), "success": r["success"]} for r in ep_rows]
        curves += [{"model": name, **c} for c in handoff_curve(eps, taus)]
    write_csv(res / "rq3_handoff.csv", curves)
    (res / "rq3_loo.json").write_text(json.dumps({**loo, "signal": args.signal,
                                                  "n_train_steps": len(train)}, indent=1))


if __name__ == "__main__":
    main()
