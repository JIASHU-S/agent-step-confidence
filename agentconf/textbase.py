"""Text-only predictors of step correctness, for the visible-context control in RQ1.

Every predictor sees only what is in the agent's context when the action has been written: the question, earlier
steps (thought, action) with the observations they returned, and the current thought and action. The current
observation, gold answers and supporting-paragraph flags are never used.

Candidates (each a logistic regression; the inverse regularisation C is chosen by question-grouped 5-fold AUROC on
the training split):
    handcrafted  the 14 features of scripts/16_review_checks.py
    tfidf        word 1-2 gram TF-IDF of the question, the history and the current step (separate vocabularies)
                 plus the handcrafted features
    embedding    sentence embeddings of the question, history and current step, their cosine similarities, the
                 largest similarity of the current step to any earlier observation, plus the handcrafted features
`select_and_fit` picks the candidate with the best grouped out-of-fold AUROC on the training split only.
"""
import hashlib
import re
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0, 10.0)
CANDIDATES = ("handcrafted", "tfidf", "embedding")


def contexts(steps, questions):
    """Per step: question, history (earlier steps and their observations) and the current step text."""
    by_q = defaultdict(list)
    for i, s in enumerate(steps):
        by_q[s["qid"]].append(i)
    out = [None] * len(steps)
    for qid, idx in by_q.items():
        idx.sort(key=lambda i: steps[i]["t"])
        hist, prev_obs = [], []
        for i in idx:
            s = steps[i]
            out[i] = {"question": questions[qid], "history": "\n".join(hist), "current": s.get("text", ""),
                      "prev_obs": list(prev_obs)}
            hist.append(f"{s.get('text', '')}\nObservation {s['t']}: {s.get('obs', '')}")
            prev_obs.append(s.get("obs", "") or "")
    return out


class HashEncoder:
    """Deterministic bag-of-words random projection. Used only by synthetic tests (no model download)."""

    def __init__(self, dim=64):
        self.dim = dim
        self.name, self.revision = f"hash:{dim}", "none"

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", (t or "").lower()):
                v = int(hashlib.sha256(w.encode()).hexdigest()[:8], 16)
                out[i, v % self.dim] += 1.0 if (v >> 8) % 2 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class HFEncoder:
    """Mean-pooled, L2-normalised sentence embeddings from a pinned Hugging Face encoder (CPU)."""

    def __init__(self, name, revision, batch_size=64, max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name, self.revision = name, revision
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision).eval()
        self.torch, self.batch_size, self.max_length = torch, batch_size, max_length

    def encode(self, texts):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok([t or "" for t in texts[i:i + self.batch_size]], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 1), np.float32)


def resolve_encoder(spec):
    """'hash:<dim>' for tests, else '<hf name>@<revision>'. Returns an encoder object."""
    if spec.startswith("hash:"):
        return HashEncoder(int(spec.split(":", 1)[1]))
    name, _, revision = spec.partition("@")
    if not revision:
        raise ValueError("A Hugging Face encoder must be pinned as name@revision")
    return HFEncoder(name, revision)


def pin_encoder(name):
    """Resolve a Hugging Face model name to name@<commit sha> (needs network); hash encoders pass through."""
    if name.startswith("hash:") or "@" in name:
        return name
    from huggingface_hub import HfApi

    return f"{name}@{HfApi().model_info(name).sha}"


def embedding_features(ctx, encoder):
    q = encoder.encode([c["question"] for c in ctx])
    h = encoder.encode([c["history"] or "(none)" for c in ctx])
    cur = encoder.encode([c["current"] for c in ctx])
    flat = [o for c in ctx for o in c["prev_obs"]]
    obs = encoder.encode(flat) if flat else np.zeros((0, cur.shape[1]), np.float32)
    best, k = np.zeros(len(ctx), np.float32), 0
    for i, c in enumerate(ctx):
        n = len(c["prev_obs"])
        if n:
            best[i] = float((obs[k:k + n] @ cur[i]).max())
        k += n
    sims = np.column_stack([(q * cur).sum(1), (h * cur).sum(1), (q * h).sum(1), best])
    return np.hstack([q, h, cur, sims]).astype(np.float32)


class _Model:
    def __init__(self, kind, C):
        self.kind, self.C = kind, C

    def fit(self, ctx, hand, y):
        if self.kind == "tfidf":
            self.vecs = {k: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
                         for k in ("question", "history", "current")}
            blocks = [self.vecs[k].fit_transform([c[k] for c in ctx]) for k in self.vecs]
        self.scaler = StandardScaler().fit(hand if self.kind != "embedding" else np.hstack([self.emb_tr, hand]))
        X = self._X(ctx, hand, train=True, blocks=blocks if self.kind == "tfidf" else None)
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit(X, y)
        return self

    def _X(self, ctx, hand, train=False, blocks=None, emb=None):
        if self.kind == "handcrafted":
            return self.scaler.transform(hand)
        if self.kind == "tfidf":
            if blocks is None:
                blocks = [self.vecs[k].transform([c[k] for c in ctx]) for k in self.vecs]
            return sparse.hstack(blocks + [sparse.csr_matrix(self.scaler.transform(hand))]).tocsr()
        e = self.emb_tr if train else emb
        return self.scaler.transform(np.hstack([e, hand]))

    def predict(self, ctx, hand, emb=None):
        return self.lr.predict_proba(self._X(ctx, hand, emb=emb))[:, 1]


def _fit(kind, C, ctx, hand, y, emb):
    m = _Model(kind, C)
    m.emb_tr = emb
    return m.fit(ctx, hand, y)


def oof_auroc(kind, C, ctx, hand, y, groups, emb=None, n_splits=5):
    from .metrics import auroc

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(n_splits, len(set(groups)))).split(hand, y, groups):
        m = _fit(kind, C, [ctx[i] for i in tr], hand[tr], y[tr], None if emb is None else emb[tr])
        oof[te] = m.predict([ctx[i] for i in te], hand[te], None if emb is None else emb[te])
    return auroc(oof, y)


def select_and_fit(ctx_tr, hand_tr, y_tr, groups_tr, ctx_te, hand_te, encoder):
    """Grouped-CV choice of C per candidate, then of the candidate; returns scores for every candidate on the
    target and the selection record. Only training labels are read."""
    emb_tr = embedding_features(ctx_tr, encoder)
    emb_te = embedding_features(ctx_te, encoder)
    record, scores = {}, {}
    for kind in CANDIDATES:
        emb = emb_tr if kind == "embedding" else None
        cv = {C: oof_auroc(kind, C, ctx_tr, hand_tr, y_tr, groups_tr, emb) for C in C_GRID}
        C = max(cv, key=lambda c: np.nan_to_num(cv[c], nan=-1))
        m = _fit(kind, C, ctx_tr, hand_tr, y_tr, emb)
        scores[kind] = m.predict(ctx_te, hand_te, emb_te if kind == "embedding" else None)
        record[kind] = {"C": C, "oof_auroc_train": cv[C], "cv": {str(k): v for k, v in cv.items()}}
    best = max(CANDIDATES, key=lambda k: np.nan_to_num(record[k]["oof_auroc_train"], nan=-1))
    return scores, {"selected": best, "candidates": record, "encoder": f"{encoder.name}@{encoder.revision}"}
