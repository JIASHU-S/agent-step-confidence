"""Bayesian step-correctness models and step -> trajectory aggregation (RQ3).

M0 (independent steps)   logit P(y_t = 1) = a[kind] + b[kind] * s_t
M1 (coupled errors)      logit P(y_t = 1) = a[kind] + b[kind] * s_t + g * e_{t-1}

s_t is a step signal on the logit scale (e.g. out-of-fold probe score), e_{t-1} = 1 - y_{t-1} marks an error at the
previous step (e_0 = 0). Comparing M0 and M1 with PSIS-LOO tests whether step errors are coupled, the same
DeltaELPD / SE evidence ratio as in the count-time-series paper.

Trajectory confidence = P(final Finish step is correct). Under M1 the previous step's correctness is unknown at test
time, so it is marginalised with a two-state forward recursion:
    q_0 = 1,  q_t = q_{t-1} * sigmoid(eta_t) + (1 - q_{t-1}) * sigmoid(eta_t + g)
"""
import numpy as np

KINDS = ["Search", "Finish", "invalid"]


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def design(steps):
    """steps: list of dicts with qid, t, kind, s, correct (sorted by qid, t). Adds prev_err."""
    rows, prev = [], {}
    for r in sorted(steps, key=lambda r: (r["qid"], r["t"])):
        e = prev.get(r["qid"], 0)
        rows.append({**r, "kind_idx": KINDS.index(r["kind"]), "prev_err": e})
        prev[r["qid"]] = 1 - int(r["correct"])
    return rows


def fit(rows, coupled, draws=1000, tune=1000, chains=4, seed=0):
    import pymc as pm

    k = np.array([r["kind_idx"] for r in rows])
    s = np.array([r["s"] for r in rows], float)
    y = np.array([r["correct"] for r in rows], int)
    e = np.array([r["prev_err"] for r in rows], float)
    with pm.Model():
        a = pm.Normal("a", 0, 2, shape=len(KINDS))
        b = pm.Normal("b", 0, 2, shape=len(KINDS))
        eta = a[k] + b[k] * s
        if coupled:
            g = pm.Normal("g", 0, 2)
            eta = eta + g * e
        pm.Bernoulli("y", logit_p=eta, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": True})
    return idata


def compare(idata0, idata1):
    """PSIS-LOO comparison. Returns elpd for both models, the difference (M1 - M0), its SE and the ratio."""
    import arviz as az

    l0, l1 = az.loo(idata0, pointwise=True), az.loo(idata1, pointwise=True)
    p0, p1 = _pointwise(l0), _pointwise(l1)
    d = p1 - p0
    delta, se = float(d.sum()), float(np.sqrt(len(d)) * d.std(ddof=1))
    return {"elpd_M0": float(p0.sum()), "elpd_M1": float(p1.sum()), "delta_elpd": delta, "se": se,
            "SR": delta / se if se > 0 else float("nan"),
            "pareto_k_max_M0": float(np.max(_pareto_k(l0))), "pareto_k_max_M1": float(np.max(_pareto_k(l1)))}


def _pointwise(loo):
    v = getattr(loo, "loo_i", None)
    if v is None:
        v = getattr(loo, "elpd_i")
    return np.asarray(v).ravel()


def _pareto_k(loo):
    return np.asarray(loo.pareto_k).ravel()


def posterior_dataset(idata):
    """idata.posterior as an xarray Dataset (newer ArviZ returns a DataTree)."""
    post = idata.posterior
    return post.to_dataset() if hasattr(post, "to_dataset") else post


def posterior(idata, max_draws=500):
    post = posterior_dataset(idata).stack(sample=("chain", "draw"))
    n = post.sizes["sample"]
    sel = np.linspace(0, n - 1, min(max_draws, n)).astype(int)
    a = post["a"].values[:, sel].T  # [S, K]
    b = post["b"].values[:, sel].T
    g = post["g"].values[sel] if "g" in post else np.zeros(len(sel))
    return a, b, g


def trajectory_confidences(episode_steps, params0, params1):
    """
    episode_steps: list of step dicts for one finished episode (sorted by t; last step is Finish).
    Returns per-method confidences that the final answer is correct, plus M1's running marginal q_t.
    """
    a0, b0, _ = params0
    a1, b1, g1 = params1
    k = np.array([KINDS.index(r["kind"]) for r in episode_steps])
    s = np.array([r["s"] for r in episode_steps], float)
    p0 = sigmoid(a0[:, k] + b0[:, k] * s)  # [S, T] independent-step probabilities
    eta1 = a1[:, k] + b1[:, k] * s
    q = np.ones(len(g1))
    running = []
    for t in range(len(k)):
        q = q * sigmoid(eta1[:, t]) + (1 - q) * sigmoid(eta1[:, t] + g1)
        running.append(float(q.mean()))
    return {
        "M0_last": float(p0[:, -1].mean()),
        "M0_product": float(np.prod(p0, axis=1).mean()),
        "M0_min": float(p0.min(axis=1).mean()),
        "M1_forward": running[-1],
        "running": running,
    }


def handoff_curve(episodes, thresholds):
    """
    episodes: list of {"running": [q_1..q_T], "success": 0/1}. Policy: hand the episode to a person at the first
    step whose running confidence falls below tau. Returns coverage (kept by the agent), accuracy of kept
    episodes, and the mean step at which hand-offs happen.
    """
    out = []
    for tau in thresholds:
        kept, steps = [], []
        for ep in episodes:
            below = [i for i, q in enumerate(ep["running"]) if q < tau]
            if below:
                steps.append(below[0] + 1)
            else:
                kept.append(ep["success"])
        out.append({"tau": float(tau), "coverage": len(kept) / len(episodes),
                    "kept_accuracy": float(np.mean(kept)) if kept else float("nan"),
                    "mean_handoff_step": float(np.mean(steps)) if steps else float("nan")})
    return out
