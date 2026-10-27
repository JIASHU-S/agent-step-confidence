"""Linear probes on hidden states: standardise -> PCA -> logistic regression."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc


def make_probe(n_components=64, C=1.0):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=0),
                         LogisticRegression(C=C, max_iter=2000))


def _fit(X, y, n_components, C):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    return make_probe(n_components, C).fit(X, y)


def oof_scores(X, y, groups, n_splits=5, n_components=64, C=1.0):
    """Out-of-fold probe probabilities, folds split by question so steps of one episode never straddle folds."""
    X, y = np.asarray(X, np.float32), np.asarray(y, int)
    out = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(n_splits, len(set(groups)))).split(X, y, groups):
        if len(set(y[tr])) < 2:
            out[te] = y[tr].mean()
            continue
        out[te] = _fit(X[tr], y[tr], n_components, C).predict_proba(X[te])[:, 1]
    return out


def fit_predict(X_tr, y_tr, X_te, n_components=64, C=1.0):
    model = _fit(np.asarray(X_tr, np.float32), np.asarray(y_tr, int), n_components, C)
    return model.predict_proba(np.asarray(X_te, np.float32))[:, 1]


def layer_sweep(F, y, groups, layers, **kw):
    """F: [n, L, d]. Returns {layer: out-of-fold AUROC} for choosing the layer on training data only."""
    return {layer: auroc(oof_scores(F[:, j, :], y, groups, **kw), y) for j, layer in enumerate(layers)}
