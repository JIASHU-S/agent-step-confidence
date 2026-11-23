"""Which kinds of step error does a TriviaQA-trained probe transfer to, and why? (CPU; run after 03 and 16.)

    python scripts/18_transfer_decomposition.py --run runs/qwen7b

EXPLORATORY. Everything here is computed on the legacy test split, which earlier analyses have already used. The
results generate hypotheses for a pre-registered confirmation on new data; they do not confirm anything themselves.

Error types follow the environment's label rule (agentconf/errtypes.py): a wrong Search is a repeat (paragraph
already retrieved in this episode), a distractor (new paragraph outside the supporting set) or a no-match; a wrong
Finish has token F1 below the threshold.

1. Decomposition -> transfer_types.csv. Signals: the in-domain probe (agent training labels, read-out chosen on them
   as in 03), the TriviaQA probe at that read-out, the TriviaQA probe at the read-out chosen on TriviaQA alone (as in
   16), and step index (score = -t). AUROC on each contrast (all steps, Search, Finish, correct Search vs each Search
   error type), pooled and over same-step pairs only (correct/wrong pairs with equal t, which removes position).
   95% CIs resample test questions; gaps to the in-domain probe use the same replicates.
2. Directions -> transfer_directions.csv. At the in-domain read-out, each probe's weight vector mapped back to
   activation space (scaler and PCA undone). Cosines between the TriviaQA direction and in-domain directions trained
   on each contrast, and among those. sub_lo/sub_hi: 2.5-97.5 percentile over refits on 80% subsamples (without
   replacement) of agent training questions and TriviaQA items -- a stability range, not a CI for the full-data
   value. A label-permutation null gives the chance level of |cosine|.
3. Removal -> transfer_inlp.csv. Iterative nullspace projection (Ravfogel et al., 2020) removes directions that
   predict TriviaQA correctness until a probe on held-out TriviaQA answers is near chance. The in-domain probe is then
   refit on the projected agent features and scored per contrast; the same number of random directions is removed as
   a control. If the shared QA-correctness component carries the answer-step signal but not the retrieval signal,
   removal should cost more on Finish than on correct-vs-distractor Search.

Writes to <run>/results/: transfer_types.csv, transfer_directions.csv, transfer_inlp.csv, transfer_decomposition.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.errtypes import auroc_pairs, error_types  # noqa: E402
from agentconf.io import read_jsonl  # noqa: E402
from agentconf.metrics import auroc  # noqa: E402
from agentconf.probes import _fit, fit_predict, oof_scores  # noqa: E402
from agentconf.runs import load_features, load_run  # noqa: E402

CONTRASTS = {  # name -> step types included (positive class = correct)
    "all": None,
    "search": ("search_ok", "search_repeat", "search_distractor", "search_nomatch"),
    "finish": ("finish_ok", "finish_wrong"),
    "search_ok_vs_repeat": ("search_ok", "search_repeat"),
    "search_ok_vs_distractor": ("search_ok", "search_distractor"),
    "search_ok_vs_nomatch": ("search_ok", "search_nomatch"),
}
DIRECTION_CONTRASTS = ("all", "search", "finish", "search_ok_vs_repeat", "search_ok_vs_distractor")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def contrast_mask(types, name):
    members = CONTRASTS[name]
    return np.ones(len(types), bool) if members is None else np.isin(types, members)


def source_only_readout(run, layers, pca, review_path):
    """Read-out chosen on TriviaQA alone: reuse 16's choice if present, else repeat its selection."""
    if review_path.exists():
        sel = json.loads(review_path.read_text(encoding="utf-8")).get("rq2_source_only", {}).get("selected")
        if sel:
            return sel["pooling"], int(sel["layer"])
    tv = np.load(run / "trivia_feats.npz")
    ty = np.array([r["correct"] for r in read_jsonl(run / "trivia.jsonl")], int)
    best = max(((p, l, auroc(oof_scores(tv[p][:, j, :], ty, np.arange(len(ty)), n_components=pca), ty))
                for p in ("mean", "last") for j, l in enumerate(layers)), key=lambda r: np.nan_to_num(r[2], nan=-1))
    return best[0], best[1]


def bootstrap_table(scores, ref, y, t, q, types, n_boot, seed=0):
    """AUROC per signal x contrast x {pooled, same_step}, question-bootstrap CIs, paired gaps to `ref`."""
    uq = np.unique(q)
    members = [np.flatnonzero(q == u) for u in uq]
    masks = {c: contrast_mask(types, c) for c in CONTRASTS}
    masks = {c: m for c, m in masks.items() if (m & (y == 0)).any() and (m & (y == 1)).any()}
    versions = {"pooled": None, "same_step": t}

    def evaluate(idx):
        out = {}
        for c, m in masks.items():
            sub = idx[m[idx]]
            for v, strata in versions.items():
                st = None if strata is None else strata[sub]
                for name, s in scores.items():
                    if v == "same_step" and name == "step_index":
                        continue  # constant within a step: 0.5 by construction
                    out[(c, v, name)] = auroc_pairs(s[sub], y[sub], st)
        return out

    point = evaluate(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uq), len(uq))])
        reps.append(evaluate(idx))
    rows = []
    for (c, v, name), a in point.items():
        m = masks[c]
        vals = np.array([r[(c, v, name)] for r in reps])
        row = {"contrast": c, "version": v, "signal": name, "n": int(m.sum()), "n_wrong": int((m & (y == 0)).sum()),
               "auroc": a, "ci_lo": np.nanpercentile(vals, 2.5), "ci_hi": np.nanpercentile(vals, 97.5)}
        if name != ref and (c, v, ref) in point:
            gaps = vals - np.array([r[(c, v, ref)] for r in reps])
            row.update({f"gap_vs_{ref}": a - point[(c, v, ref)], "gap_lo": np.nanpercentile(gaps, 2.5),
                        "gap_hi": np.nanpercentile(gaps, 97.5)})
        rows.append(row)
    return rows


def raw_direction(X, y, pca):
    """Unit weight vector of a standardise -> PCA -> logistic probe, expressed in activation space."""
    model = _fit(np.asarray(X, np.float32), np.asarray(y, int), pca, 1.0)
    sc, pc, lr = (model.named_steps[k] for k in ("standardscaler", "pca", "logisticregression"))
    w = (pc.components_.T @ lr.coef_[0]) / sc.scale_
    return w / np.linalg.norm(w)


def directions(Xa, ya, qa, types_a, Xs, ys, pca, n_boot, n_perm, seed=0):
    names = ["trivia"] + list(DIRECTION_CONTRASTS)

    def fit_all(ia, isrc):
        d = {"trivia": raw_direction(Xs[isrc], ys[isrc], pca)}
        for c in DIRECTION_CONTRASTS:
            m = contrast_mask(types_a[ia], c)
            d[c] = raw_direction(Xa[ia][m], ya[ia][m], pca)
        return d

    def cosines(d):
        return {(a, b): float(d[a] @ d[b]) for i, a in enumerate(names) for b in names[i + 1:]}

    point = cosines(fit_all(np.arange(len(ya)), np.arange(len(ys))))
    rng = np.random.default_rng(seed)
    uq = np.unique(qa)
    members = [np.flatnonzero(qa == u) for u in uq]
    reps = []  # 80% subsamples without replacement: duplicated rows would change the PCA and bias the cosines
    for _ in range(n_boot):
        ia = np.concatenate([members[i] for i in rng.choice(len(uq), int(0.8 * len(uq)), replace=False)])
        reps.append(cosines(fit_all(ia, rng.choice(len(ys), int(0.8 * len(ys)), replace=False))))
    w_src = raw_direction(Xs, ys, pca)
    null = [abs(float(w_src @ raw_direction(Xa, rng.permutation(ya), pca))) for _ in range(n_perm)]
    rows = []
    for pair, c in point.items():
        vals = np.array([r[pair] for r in reps])
        rows.append({"a": pair[0], "b": pair[1], "cosine": c, "sub_lo": np.percentile(vals, 2.5),
                     "sub_hi": np.percentile(vals, 97.5)})
    rows.append({"a": "trivia", "b": "permuted agent labels (|cos|, 95th pct)", "cosine": float(np.percentile(null, 95)),
                 "sub_lo": float(np.min(null)), "sub_hi": float(np.max(null))})
    return rows


def inlp(Z, ys, pca, target, max_iter, seed=0):
    """Directions (orthonormal columns, standardised space) removing linear TriviaQA-correctness information."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ys))
    tr, va = order[: int(0.8 * len(ys))], order[int(0.8 * len(ys)):]
    Q = np.zeros((Z.shape[1], 0), np.float32)
    trace = []
    for _ in range(max_iter):
        Zp = Z - (Z @ Q) @ Q.T
        n_comp = min(pca, len(tr) - 1)
        clf = make_pipeline(PCA(n_components=n_comp, random_state=0), LogisticRegression(max_iter=2000))
        clf.fit(Zp[tr], ys[tr])
        val = auroc(clf.predict_proba(Zp[va])[:, 1], ys[va])
        trace.append(val)
        if val < target:
            break
        w = clf.named_steps["pca"].components_.T @ clf.named_steps["logisticregression"].coef_[0]
        w = w - Q @ (Q.T @ w)
        Q = np.hstack([Q, (w / np.linalg.norm(w)).astype(np.float32)[:, None]])
    return Q, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-dir", type=int, default=100, help="refits per bootstrap for direction cosines")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--inlp-target", type=float, default=0.55, help="stop when held-out TriviaQA AUROC falls below")
    ap.add_argument("--inlp-max", type=int, default=40)
    ap.add_argument("--n-random", type=int, default=5, help="random-removal control draws")
    args = ap.parse_args()
    res = args.run / "results"

    meta, episodes, steps = load_run(args.run)
    layers = meta["layers"]
    y = np.array([s["correct"] for s in steps], int)
    t = np.array([s["t"] for s in steps], int)
    q = np.array([s["qid"] for s in steps])
    types = error_types(steps)
    tr = np.array([s["split"] == "train" for s in steps])
    te = ~tr
    choice = json.loads((res / "probe_choice.json").read_text(encoding="utf-8"))
    agent_ro = (choice["pooling"], int(choice["layer"]))
    source_ro = source_only_readout(args.run, layers, args.pca, res / "review_checks.json")
    tv = np.load(args.run / "trivia_feats.npz")
    ys = np.array([r["correct"] for r in read_jsonl(args.run / "trivia.jsonl")], int)
    feats = {p: load_features(args.run, steps, p) for p in {agent_ro[0], source_ro[0]}}

    def X(ro, rows=slice(None)):
        return feats[ro[0]][rows, layers.index(ro[1]), :]

    def S(ro):
        return tv[ro[0]][:, layers.index(ro[1]), :]

    counts = {split: {k: int(v) for k, v in zip(*np.unique(types[m], return_counts=True))}
              for split, m in (("train", tr), ("test", te))}
    print(f"read-outs: in-domain {agent_ro}, TriviaQA-only {source_ro}; test types {counts['test']}")

    # ---- 1. decomposition
    scores = {
        "in_domain": fit_predict(X(agent_ro, tr), y[tr], X(agent_ro, te), n_components=args.pca),
        "transfer_at_in_domain_readout": fit_predict(S(agent_ro), ys, X(agent_ro, te), n_components=args.pca),
        "transfer_at_trivia_readout": fit_predict(S(source_ro), ys, X(source_ro, te), n_components=args.pca),
        "step_index": -t[te].astype(float),
    }
    rows = bootstrap_table(scores, "in_domain", y[te], t[te], q[te], types[te], args.n_boot)
    write_csv(res / "transfer_types.csv", rows)
    for r in rows:
        if r["version"] == "same_step" or r["contrast"] in ("search_ok_vs_repeat", "search_ok_vs_distractor", "finish"):
            print(f"  {r['contrast']:24s} {r['version']:9s} {r['signal']:30s} {r['auroc']:.3f} "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']} wrong={r['n_wrong']}")

    # ---- 2. directions (in-domain read-out, agent training split only)
    dir_rows = directions(X(agent_ro, tr), y[tr], q[tr], types[tr], S(agent_ro), ys, args.pca,
                          args.n_boot_dir, args.n_perm)
    write_csv(res / "transfer_directions.csv", dir_rows)
    for r in dir_rows:
        print(f"  cos({r['a']}, {r['b']}) = {r['cosine']:.3f} [{r['sub_lo']:.3f}, {r['sub_hi']:.3f}]")

    # ---- 3. INLP removal of TriviaQA-correctness directions
    sc = StandardScaler().fit(S(agent_ro))
    Zs = sc.transform(S(agent_ro)).astype(np.float32)
    Za = sc.transform(X(agent_ro)).astype(np.float32)
    Q, trace = inlp(Zs, ys, args.pca, args.inlp_target, args.inlp_max)
    k = Q.shape[1]
    print(f"INLP: removed {k} directions; held-out TriviaQA AUROC per iteration {np.round(trace, 3).tolist()}")

    def refit(Zagent):
        return fit_predict(Zagent[tr], y[tr], Zagent[te], n_components=args.pca)

    rng = np.random.default_rng(1)
    rand_scores = []
    for _ in range(args.n_random):
        R, _ = np.linalg.qr(rng.standard_normal((Za.shape[1], k)).astype(np.float32))
        rand_scores.append(refit(Za - (Za @ R) @ R.T))
    Za_inlp = Za - (Za @ Q) @ Q.T
    inlp_scores = {
        "in_domain": refit(Za),
        "in_domain_after_inlp": refit(Za_inlp),
        # random control: average of per-draw logits is not a probe; use the median-AUROC draw as one representative
        "in_domain_after_random": rand_scores[int(np.argsort([auroc(s, y[te]) for s in rand_scores])[len(rand_scores) // 2])],
        "transfer_after_inlp": fit_predict(Zs - (Zs @ Q) @ Q.T, ys, Za_inlp[te], n_components=args.pca),
    }
    inlp_rows = bootstrap_table(inlp_scores, "in_domain", y[te], t[te], q[te], types[te], args.n_boot)
    for r in inlp_rows:
        r["k_removed"] = k
    write_csv(res / "transfer_inlp.csv", inlp_rows)
    for r in inlp_rows:
        if r["version"] == "pooled" and r["signal"] != "in_domain":
            print(f"  {r['contrast']:24s} {r['signal']:24s} {r['auroc']:.3f}  gap {r.get('gap_vs_in_domain', float('nan')):+.3f} "
                  f"[{r.get('gap_lo', float('nan')):+.3f}, {r.get('gap_hi', float('nan')):+.3f}]")

    summary = {"exploratory": "legacy test split, already used in earlier analyses; hypothesis generation only",
               "readout_in_domain": agent_ro, "readout_trivia_only": source_ro, "type_counts": counts,
               "inlp": {"k_removed": k, "heldout_trivia_auroc_trace": trace, "target": args.inlp_target},
               "random_control_auroc_all": [auroc(s, y[te]) for s in rand_scores],
               "n_boot": args.n_boot, "n_boot_dir": args.n_boot_dir, "n_perm": args.n_perm, "pca": args.pca}
    (res / "transfer_decomposition.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
