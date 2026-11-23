"""Held-out HotpotQA question selection (protocols/source_swap_v1.md section 11.3; submission_v2 new calibration/final).

The rule reads question IDs and texts only, never answers, labels or model outputs:
  1. pool = the pinned HotpotQA distractor validation file;
  2. drop the legacy IDs (600 train + 300 test), exact normalised-text repeats inside the pool (keep the first by
     hash order), exact normalised-text matches of any reference question (legacy targets and source-domain
     questions), and near duplicates: TF-IDF (word 1-2 grams over normalised text) cosine >= NEAR_DUP_COSINE to any
     reference question;
  3. order the rest by sha256(SALT + id); the first sizes[0] go to the first split, the next sizes[1] to the next...
"""
import hashlib

import numpy as np

from .text import normalize

SALT = "heldout-v1:"
NEAR_DUP_COSINE = 0.8


def hash_order(ids, salt=SALT):
    return sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())


def max_cosine(queries, references, chunk=2000):
    """Max TF-IDF cosine of each query against all references (vectoriser fitted on both sets)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not len(queries) or not len(references):
        return np.zeros(len(queries))
    vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit(list(queries) + list(references))
    Q, R = vec.transform(queries), vec.transform(references).T.tocsc()
    out = np.zeros(Q.shape[0])
    for a in range(0, Q.shape[0], chunk):
        out[a:a + chunk] = (Q[a:a + chunk] @ R).max(axis=1).toarray().ravel()
    return out


def select_heldout(pool, exclude_ids, reference_questions, sizes, salt=SALT, cosine=NEAR_DUP_COSINE):
    """pool: list of (id, question). Returns (splits: list of id lists, report: exclusion counts)."""
    exclude_ids = set(exclude_ids)
    ref_norm = [normalize(q) for q in reference_questions]
    ref_set = set(ref_norm)
    text = {i: normalize(q) for i, q in pool}
    if len(text) != len(pool):
        raise ValueError("duplicate IDs in the pool")
    report = {"pool": len(pool), "legacy_id": 0, "pool_text_repeat": 0, "reference_exact": 0, "reference_near": 0}
    seen, kept = set(), []
    for i in hash_order(text, salt):
        if i in exclude_ids:
            report["legacy_id"] += 1
        elif text[i] in seen:
            report["pool_text_repeat"] += 1
        elif text[i] in ref_set:
            report["reference_exact"] += 1
            seen.add(text[i])
        else:
            seen.add(text[i])
            kept.append(i)
    near = max_cosine([text[i] for i in kept], ref_norm) >= cosine
    report["reference_near"] = int(near.sum())
    kept = [i for i, n in zip(kept, near) if not n]
    report["eligible"] = len(kept)
    if sum(sizes) > len(kept):
        raise ValueError(f"only {len(kept)} eligible questions, {sum(sizes)} requested")
    splits, a = [], 0
    for n in sizes:
        splits.append(kept[a:a + n])
        a += n
    return splits, report
