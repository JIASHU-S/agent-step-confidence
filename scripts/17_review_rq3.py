"""Review follow-up for RQ3 (CPU, seconds; run after 03, 04 and 06).

    python scripts/17_review_rq3.py --run runs/qwen7b

1. Hand-off with a matched aggregation rule. Fig. 2 compared M0's *running product* with M1's forward marginal, so
   the gap mixed the history term with the aggregation rule. A third policy hands off at the first step whose M0
   *current-step* probability falls below tau. M0 product and M1 forward are the running values saved by 04
   (full posterior); M0 current step uses M0's posterior-mean parameters from rq3_params.csv (plug-in). An episode is
   kept iff the minimum of the running quantity over its steps is >= tau, as in 06. Writes review_handoff_bands.csv
   and review_handoff.json (area under the risk-coverage curve with episode-bootstrap CIs).
2. Is the previous-error effect state dependence or question heterogeneity? M0/M1 are refitted on the training
   split by MAP with the same N(0, 2^2) priors, (a) as in 04, (b) with a Gaussian question random intercept
   integrated out by 40-node Gauss-Hermite quadrature, and (c) as (b) plus step index (one-hot, t >= 4 pooled) and a
   repeated-query indicator. Reports g with a 90% Laplace interval, sigma_q, and the log-likelihood gain of M1.
   Writes review_rq3_sensitivity.json. With --nuts the same three pairs are refitted with PyMC/NUTS and compared by
   PSIS-LOO (review_rq3_nuts.json); run through scripts/python_local.ps1 so PyTensor finds g++.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.metrics import aurc  # noqa: E402
from agentconf.runs import load_run  # noqa: E402
from agentconf.seqmodel import KINDS, design, logit  # noqa: E402


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def repeat_flags(steps):
    prev, out = defaultdict(set), {}
    for s in sorted(steps, key=lambda s: (s["qid"], s["t"])):
        a = norm(s.get("arg", ""))
        out[(s["qid"], s["t"])] = int(s["kind"] == "Search" and a in prev[s["qid"]])
        if s["kind"] == "Search":
            prev[s["qid"]].add(a)
    return out


def handoff_stats(min_conf, success, taus):
    return [(float((min_conf >= t).mean()), float(success[min_conf >= t].mean()) if (min_conf >= t).any()
             else np.nan) for t in taus]


# ---------------------------------------------------------------- MAP fits with an optional random intercept
class Model:
    def __init__(self, rows, coupled, random_intercept, covariates, n_nodes=40):
        self.k = np.array([r["kind_idx"] for r in rows])
        self.s = np.array([r["s"] for r in rows], float)
        self.y = np.array([r["correct"] for r in rows], float)
        self.e = np.array([r["prev_err"] for r in rows], float)
        self.X = np.array([[r[c] for c in covariates] for r in rows], float).reshape(len(rows), len(covariates))
        qids = sorted({r["qid"] for r in rows})
        self.qi = np.array([qids.index(r["qid"]) for r in rows])
        self.nq = len(qids)
        self.coupled, self.ri, self.nc = coupled, random_intercept, len(covariates)
        x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
        self.nodes, self.logw = x, np.log(w / w.sum())
        self.names = ([f"a[{k}]" for k in KINDS] + [f"b[{k}]" for k in KINDS] + [f"beta[{c}]" for c in covariates]
                      + (["g"] if coupled else []) + (["log_sigma_q"] if random_intercept else []))

    def unpack(self, th):
        K, i = len(KINDS), 0
        a, b = th[:K], th[K:2 * K]
        i = 2 * K
        beta = th[i:i + self.nc]
        i += self.nc
        g = th[i] if self.coupled else 0.0
        i += int(self.coupled)
        ls = th[i] if self.ri else None
        return a, b, beta, g, ls

    def loglik(self, th):
        a, b, beta, g, ls = self.unpack(th)
        eta = a[self.k] + b[self.k] * self.s + self.X @ beta + g * self.e
        if not self.ri:
            return float(np.sum(self.y * log_expit(eta) + (1 - self.y) * log_expit(-eta)))
        sig = np.exp(ls)
        E = eta[:, None] + sig * self.nodes[None, :]  # [n, nodes]
        ll = self.y[:, None] * log_expit(E) + (1 - self.y[:, None]) * log_expit(-E)
        per_q = np.zeros((self.nq, len(self.nodes)))
        np.add.at(per_q, self.qi, ll)
        return float(np.sum(logsumexp(per_q + self.logw[None, :], axis=1)))

    def neg_log_post(self, th):
        a, b, beta, g, ls = self.unpack(th)
        prior = (np.sum(a ** 2) + np.sum(b ** 2) + np.sum(beta ** 2) + (g ** 2 if self.coupled else 0.0)) / (2 * 4.0)
        if self.ri:  # half-normal(1) prior on sigma, written on log sigma (with Jacobian)
            prior += np.exp(2 * ls) / 2 - ls
        return -self.loglik(th) + prior

    def fit(self):
        th0 = np.zeros(len(self.names))
        if self.ri:
            th0[-1] = np.log(0.5)
        r = minimize(self.neg_log_post, th0, method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 10000})
        r = minimize(self.neg_log_post, r.x, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 40000, "maxfev": 40000})
        H = hessian(self.neg_log_post, r.x)
        cov = np.linalg.inv(H)
        h = 1e-5
        grad = np.array([(self.neg_log_post(r.x + h * e) - self.neg_log_post(r.x - h * e)) / (2 * h)
                         for e in np.eye(len(r.x))])
        # converged = stationary point with a positive-definite Hessian
        self.theta, self.cov = r.x, cov
        self.max_grad = float(np.abs(grad).max())
        self.success = bool(self.max_grad < 1e-2 and np.all(np.linalg.eigvalsh(H) > 0))
        return self

    def summary(self):
        out = {"converged": self.success, "max_abs_grad": self.max_grad, "loglik": self.loglik(self.theta)}
        sd = np.sqrt(np.diag(self.cov))
        for n, v, s in zip(self.names, self.theta, sd):
            if n == "g" or n == "log_sigma_q" or n.startswith("beta"):
                out[n] = {"map": float(v), "q05": float(v - 1.645 * s), "q95": float(v + 1.645 * s)}
        if self.ri:
            ls = out.pop("log_sigma_q")
            out["sigma_q"] = {k: float(np.exp(v)) for k, v in ls.items()}
        return out


def hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit_nuts(rows, coupled, random_intercept, covariates, draws, tune, chains, seed, target_accept=0.9):
    """PyMC/NUTS version of the models above (same priors as 04; half-normal(1) on sigma_q; non-centred)."""
    import pymc as pm

    k = np.array([r["kind_idx"] for r in rows])
    s = np.array([r["s"] for r in rows], float)
    y = np.array([r["correct"] for r in rows], int)
    e = np.array([r["prev_err"] for r in rows], float)
    qids = sorted({r["qid"] for r in rows})
    qi = np.array([qids.index(r["qid"]) for r in rows])
    with pm.Model():
        a = pm.Normal("a", 0, 2, shape=len(KINDS))
        b = pm.Normal("b", 0, 2, shape=len(KINDS))
        eta = a[k] + b[k] * s
        if random_intercept:
            sigma = pm.HalfNormal("sigma_q", 1.0)
            z = pm.Normal("z_q", 0, 1, shape=len(qids))
            eta = eta + sigma * z[qi]
        if covariates:
            X = np.array([[r[c] for c in covariates] for r in rows], float)
            beta = pm.Normal("beta", 0, 2, shape=len(covariates))
            eta = eta + pm.math.dot(X, beta)
        if coupled:
            g = pm.Normal("g", 0, 2)
            eta = eta + g * e
        pm.Bernoulli("y", logit_p=eta, observed=y)
        return pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False,
                         target_accept=target_accept, idata_kwargs={"log_likelihood": True})


def nuts_summary(idata):
    import arviz as az

    from agentconf.seqmodel import posterior_dataset

    post = posterior_dataset(idata)
    out = {}
    for v in ("g", "sigma_q"):
        if v in post:
            x = post[v].values.ravel()
            out[v] = {"mean": float(x.mean()), "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95))}
    summ = az.summary(idata, var_names=[v for v in ("a", "b", "beta", "g", "sigma_q") if v in post])
    out["max_rhat"] = float(summ["r_hat"].max())
    out["min_ess_bulk"] = float(summ["ess_bulk"].min())
    div = idata.sample_stats["diverging"].values
    out["divergences"] = int(div.sum())
    return out


def run_nuts(train, res, draws, tune, chains, target_accept):
    from agentconf.seqmodel import compare

    out = {"n_train_steps": len(train), "draws": draws, "tune": tune, "chains": chains,
           "target_accept": target_accept}
    covs = ["t2", "t3", "t4", "repeat"]
    for name, ri, cov in [("as_in_04", False, []), ("random_intercept", True, []),
                          ("random_intercept_position_repeat", True, covs)]:
        i0 = fit_nuts(train, False, ri, cov, draws, tune, chains, seed=20, target_accept=target_accept)
        i1 = fit_nuts(train, True, ri, cov, draws, tune, chains, seed=21, target_accept=target_accept)
        out[name] = {"M1": nuts_summary(i1), "M0": nuts_summary(i0), "loo": compare(i0, i1)}
        print(name, json.dumps(out[name], indent=1), flush=True)
        (res / "review_rq3_nuts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--nuts", action="store_true", help="also refit the sensitivity models with PyMC/NUTS")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.99)
    args = ap.parse_args()
    res = args.run / "results"

    # ---- 1. hand-off with a matched aggregation rule
    params = {r["param"]: float(r["mean"]) for r in read_csv(res / "rq3_params.csv") if r["model"] == "M0"}
    a0 = np.array([params[f"a[{i}]"] for i in range(len(KINDS))])
    b0 = np.array([params[f"b[{i}]"] for i in range(len(KINDS))])
    test = defaultdict(list)
    for r in read_csv(res / "step_scores.csv"):
        if r["split"] == "test":
            test[r["qid"]].append((int(r["t"]), KINDS.index(r["kind"]), float(logit(float(r["probe"])))))
    ep = read_csv(res / "rq3_episodes.csv")
    succ = np.array([int(r["success"]) for r in ep])
    step_min = []
    for r in ep:
        st = sorted(test[r["qid"]])
        k = np.array([x[1] for x in st])
        s = np.array([x[2] for x in st])
        step_min.append(float(expit(a0[k] + b0[k] * s).min()))
    scores = {"M0_product": np.array([min(json.loads(r["running_M0"])) for r in ep]),
              "M0_step": np.array(step_min),
              "M1_forward": np.array([min(json.loads(r["running_M1"])) for r in ep])}
    taus = np.linspace(0.05, 0.95, 19)
    rng = np.random.default_rng(1)
    n = len(succ)
    curves = {m: [] for m in scores}
    pairs = [("M0_product", "M1_forward"), ("M0_step", "M1_forward"), ("M0_product", "M0_step")]
    diffs = {f"{p}_minus_{q}": [] for p, q in pairs}
    for _ in range(args.n_boot):
        i = rng.integers(0, n, n)
        for m, sc in scores.items():
            curves[m].append(handoff_stats(sc[i], succ[i], taus))
        a = {m: aurc(sc[i], succ[i]) for m, sc in scores.items()}
        for p, q in pairs:
            diffs[f"{p}_minus_{q}"].append(a[p] - a[q])
    point = {m: aurc(sc, succ) for m, sc in scores.items()}
    summary = {"n_episodes": n, "n_test_questions": len(test), "aurc": point}
    for p, q in pairs:
        v = np.asarray(diffs[f"{p}_minus_{q}"])
        summary[f"aurc_{p}_minus_{q}"] = {"point": point[p] - point[q], "ci_low": float(np.percentile(v, 2.5)),
                                          "ci_high": float(np.percentile(v, 97.5))}
    band_rows = []
    for m, sc in scores.items():
        pt = handoff_stats(sc, succ, taus)
        arr = np.array(curves[m], float)
        for j, tau in enumerate(taus):
            acc = arr[:, j, 1]
            acc = acc[np.isfinite(acc)]
            band_rows.append({"policy": m, "tau": float(tau), "coverage": pt[j][0], "kept_accuracy": pt[j][1],
                              "coverage_all_questions": pt[j][0] * n / len(test),
                              "acc_low": float(np.percentile(acc, 2.5)) if len(acc) > 50 else np.nan,
                              "acc_high": float(np.percentile(acc, 97.5)) if len(acc) > 50 else np.nan})
    write_csv(res / "review_handoff_bands.csv", band_rows)
    (res / "review_handoff.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))

    # ---- 2. random-intercept sensitivity for g
    _, _, raw_steps = load_run(args.run)
    rep = repeat_flags(raw_steps)
    steps = []
    for r in read_csv(res / "step_scores.csv"):
        if r["split"] != "train":
            continue
        t = int(r["t"])
        steps.append({"qid": r["qid"], "t": t, "kind": r["kind"], "correct": int(r["correct"]),
                      "s": float(logit(float(r["probe"]))), "repeat": rep.get((r["qid"], t), 0),
                      "t2": int(t == 2), "t3": int(t == 3), "t4": int(t >= 4)})
    train = design(steps)
    sens = {"n_train_steps": len(train), "n_train_questions": len({r["qid"] for r in train})}
    for name, ri, cov in [("as_in_04", False, []), ("random_intercept", True, []),
                          ("random_intercept_position_repeat", True, ["t2", "t3", "t4", "repeat"])]:
        m0 = Model(train, False, ri, cov).fit()
        m1 = Model(train, True, ri, cov).fit()
        s1 = m1.summary()
        sens[name] = {**s1, "loglik_gain_M1": s1["loglik"] - m0.summary()["loglik"], "M0_converged": m0.success}
    (res / "review_rq3_sensitivity.json").write_text(json.dumps(sens, indent=1), encoding="utf-8")
    print(json.dumps(sens, indent=1))
    if args.nuts:
        run_nuts(train, res, args.draws, args.tune, args.chains, args.target_accept)


if __name__ == "__main__":
    main()
