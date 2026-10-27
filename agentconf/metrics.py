import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(scores, correct):
    correct = np.asarray(correct, bool)
    if correct.all() or (~correct).all():
        return float("nan")
    return float(roc_auc_score(correct, scores))


def risk_coverage(scores, correct):
    """Risk (error among answered) as we answer questions in order of decreasing confidence.
    Ties are broken randomly but deterministically."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(len(scores)), -scores))
    err = np.cumsum(~correct[order])
    k = np.arange(1, len(scores) + 1)
    return k / len(scores), err / k


def aurc(scores, correct):
    cov, risk = risk_coverage(scores, correct)
    return float(np.mean(risk))


def acc_at_coverage(scores, correct, coverage):
    cov, risk = risk_coverage(scores, correct)
    i = max(int(np.ceil(coverage * len(cov))) - 1, 0)
    return float(1 - risk[i])


def ece(probs, correct, n_bins=10):
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(probs[m].mean() - correct[m].mean())
    return float(total)


def paired_bootstrap(metric, s1, s2, correct, n_boot=1000, seed=0):
    """CI for metric(s1) - metric(s2) under resampling questions."""
    s1, s2, correct = map(np.asarray, (s1, s2, correct))
    rng = np.random.default_rng(seed)
    n = len(correct)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d = metric(s1[i], correct[i]) - metric(s2[i], correct[i])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def brier(probs, correct):
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    return float(np.mean((probs - correct) ** 2))


def platt(train_scores, train_y, test_scores):
    """1-D logistic recalibration: maps any score to a probability using labelled training steps."""
    from sklearn.linear_model import LogisticRegression

    x = np.nan_to_num(np.asarray(train_scores, float)).reshape(-1, 1)
    y = np.asarray(train_y, int)
    if len(set(y)) < 2:
        return np.full(len(test_scores), y.mean() if len(y) else 0.5)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(x, y)
    return lr.predict_proba(np.nan_to_num(np.asarray(test_scores, float)).reshape(-1, 1))[:, 1]


def summarize(scores, probs, correct):
    return {"auroc": auroc(scores, correct), "brier": brier(probs, correct), "ece": ece(probs, correct),
            "n": int(len(correct)), "base_rate": float(np.mean(correct))}
