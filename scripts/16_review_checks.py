"""Review follow-up checks that need no refitting of the Bayesian models (CPU; run after 03, 04 and 06).

    python scripts/16_review_checks.py --run runs/qwen7b

1. Visible-text baseline. The action-token probe reads the state after the action text is written, so a logistic
   regression on what the agent can already see (step index, action type, repeated query, overlap with earlier
   queries and with titles already retrieved, query/answer length, whether a Finish answer appears in an earlier
   observation) is fitted on the training split. Probe minus this baseline, and both combined, on the test split.
2. Within-question AUROC with every eligible question weighted equally (Table 2 pools all pairs).
3. Finish-only AUROC gap probe - consistency (one Finish per question, so questions are resampled).
4. RQ3 gaps as full-sample point estimates with episode-bootstrap percentile CIs (06 printed bootstrap means).
5. RQ2 with the read-out (pooling, layer) chosen on TriviaQA only, so no agent label is used.
6. Finish steps with 0 < F1 < 1, for a manual check of the F1 >= 0.5 rule.

Writes review_checks.json and review_finish_borderline.csv to <run>/results/.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentconf.metrics import auroc, brier, ece  # noqa: E402
from agentconf.probes import fit_predict, oof_scores  # noqa: E402
from agentconf.runs import load_run  # noqa: E402

TEXT_FEATS = ["t2", "t3", "t4", "t5", "t6", "is_finish", "is_invalid", "repeat", "max_jac", "title_seen",
              "title_ov", "n_prev", "arg_len", "ans_in_obs"]


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


def text_features(steps):
    """Features visible in the agent's context when the action has been written (before it is executed)."""
    by_q = defaultdict(list)
    for s in steps:
        by_q[s["qid"]].append(s)
    out = {}
    for qid, st in by_q.items():
        st.sort(key=lambda s: s["t"])
        prev_q, titles, obs = [], [], ""
        for s in st:
            a = norm(s.get("arg", ""))
            tok = set(a.split())
            jac = [len(tok & set(p.split())) / max(len(tok | set(p.split())), 1) for p in prev_q]
            tov = [len(tok & set(norm(t).split())) / max(len(set(norm(t).split())), 1) for t in titles]
            is_search = s["kind"] == "Search"
            out[(qid, s["t"])] = {
                **{f"t{t}": int(s["t"] == t) for t in range(2, 7)},
                "is_finish": int(s["kind"] == "Finish"), "is_invalid": int(s["kind"] == "invalid"),
                "repeat": int(is_search and a in prev_q),
                "max_jac": max(jac, default=0.0) if is_search else 0.0,
                "title_seen": int(is_search and any(a == norm(t) for t in titles)),
                "title_ov": max(tov, default=0.0) if is_search else 0.0,
                "n_prev": len(prev_q), "arg_len": len(tok),
                "ans_in_obs": int(s["kind"] == "Finish" and bool(a) and a in obs),
            }
            if is_search:
                prev_q.append(a)
                if s.get("title"):
                    titles.append(s["title"])
            obs += " " + norm(s.get("obs", ""))
    return out


def boot_ci(groups, n_boot, seed, stat):
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ids, len(ids))])
        vals.append(stat(idx))
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def with_ci(groups, n_boot, stat, seed=0):
    lo, hi = boot_ci(groups, n_boot, seed, stat)
    return {"point": float(stat(np.arange(len(groups)))), "ci_low": lo, "ci_high": hi}


def within_equal(q, y, score, idx):
    """Mean over questions (with both labels) of the share of correct/wrong pairs ranked correctly."""
    per = defaultdict(list)
    for i in idx:
        per[q[i]].append(i)
    vals = []
    for members in per.values():
        m = np.array(members)
        pos, neg = score[m][y[m] == 1], score[m][y[m] == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            vals.append(((d > 0) + 0.5 * (d == 0)).mean())
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()
    res = args.run / "results"
    out = {}

    _, episodes, steps = load_run(args.run)
    tf = text_features(steps)
    rows = read_csv(res / "step_scores.csv")
    for r in rows:
        r.update(tf[(r["qid"], int(r["t"]))])
    tr = np.array([r["split"] == "train" for r in rows])
    te = ~tr
    y = np.array([int(r["correct"]) for r in rows])
    q = np.array([r["qid"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    probe = np.array([float(r["probe"]) for r in rows])
    cons = np.nan_to_num(np.array([float(r["consistency"]) if r["consistency"] not in ("", "nan") else np.nan
                                   for r in rows]))
    X = np.array([[float(r[f]) for f in TEXT_FEATS] for r in rows])
    plogit = np.log(np.clip(probe, 1e-4, 1 - 1e-4) / (1 - np.clip(probe, 1e-4, 1 - 1e-4)))

    # ---- 1. visible-text baseline
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y[tr])
    text = lr.predict_proba(X)[:, 1]
    Xc = np.column_stack([X, plogit])
    both = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xc[tr], y[tr]).predict_proba(Xc)[:, 1]
    tb = {}
    for sl in ["all", "Search", "Finish"]:
        m = te & ((kind == sl) if sl != "all" else True)
        idx = np.flatnonzero(m)
        g = q[idx]
        tb[sl] = {
            "n": int(m.sum()), "text_auroc": auroc(text[m], y[m]), "probe_auroc": auroc(probe[m], y[m]),
            "consistency_auroc": auroc(cons[m], y[m]), "text_plus_probe_auroc": auroc(both[m], y[m]),
            "probe_minus_text": with_ci(g, args.n_boot, lambda i: auroc(probe[idx][i], y[idx][i])
                                        - auroc(text[idx][i], y[idx][i])),
            "text_minus_consistency": with_ci(g, args.n_boot, lambda i: auroc(text[idx][i], y[idx][i])
                                              - auroc(cons[idx][i], y[idx][i])),
        }
    first = te & (kind == "Search") & (X[:, TEXT_FEATS.index("n_prev")] == 0)
    idx = np.flatnonzero(first)
    tb["first_search"] = {"n": int(first.sum()), "accuracy": float(y[first].mean()),
                          "probe_auroc": auroc(probe[first], y[first]),
                          "consistency_auroc": auroc(cons[first], y[first]),
                          "probe_minus_consistency": with_ci(q[idx], args.n_boot,
                                                             lambda i: auroc(probe[idx][i], y[idx][i])
                                                             - auroc(cons[idx][i], y[idx][i]))}
    rep = te & (X[:, TEXT_FEATS.index("repeat")] == 1)
    tb["repeat_queries_test"] = {"n": int(rep.sum()), "accuracy": float(y[rep].mean()) if rep.any() else None}
    out["text_baseline"] = tb

    # ---- 2. within-question AUROC, equal weight per question
    idx = np.flatnonzero(te)
    qq, yy, pp, cc = q[idx], y[idx], probe[idx], cons[idx]
    out["within_question_equal_weight"] = {
        "n_questions": int(sum(1 for g in np.unique(qq) if 0 < yy[qq == g].mean() < 1)),
        "probe": with_ci(qq, args.n_boot, lambda i: within_equal(qq[i], yy[i], pp[i], range(len(i)))),
        "consistency": with_ci(qq, args.n_boot, lambda i: within_equal(qq[i], yy[i], cc[i], range(len(i)))),
        "probe_minus_consistency": with_ci(qq, args.n_boot, lambda i: within_equal(qq[i], yy[i], pp[i], range(len(i)))
                                           - within_equal(qq[i], yy[i], cc[i], range(len(i)))),
    }

    # ---- 3. Finish-only gap
    idx = np.flatnonzero(te & (kind == "Finish"))
    out["finish_probe_minus_consistency"] = with_ci(q[idx], args.n_boot, lambda i: auroc(probe[idx][i], y[idx][i])
                                                    - auroc(cons[idx][i], y[idx][i]))

    # ---- 4. RQ3 point estimates
    ep = read_csv(res / "rq3_episodes.csv")
    succ = np.array([int(r["success"]) for r in ep])
    agg = {m: np.array([float(r[m]) for r in ep]) for m in ["M0_last", "M0_product", "M0_min", "M1_forward"]}
    eq = np.array([r["qid"] for r in ep])
    out["rq3"] = {
        "brier": {m: brier(v, succ) for m, v in agg.items()},
        "ece": {m: ece(v, succ) for m, v in agg.items()},
        "mean_confidence": {m: float(v.mean()) for m, v in agg.items()},
        "ece_product_minus_forward": with_ci(eq, args.n_boot, lambda i: ece(agg["M0_product"][i], succ[i])
                                             - ece(agg["M1_forward"][i], succ[i]), seed=1),
        "ece_last_minus_forward": with_ci(eq, args.n_boot, lambda i: ece(agg["M0_last"][i], succ[i])
                                          - ece(agg["M1_forward"][i], succ[i]), seed=1),
        "auroc_forward_minus_last": with_ci(eq, args.n_boot, lambda i: auroc(agg["M1_forward"][i], succ[i])
                                            - auroc(agg["M0_last"][i], succ[i]), seed=1),
        "brier_forward_minus_last": with_ci(eq, args.n_boot, lambda i: brier(agg["M1_forward"][i], succ[i])
                                            - brier(agg["M0_last"][i], succ[i]), seed=1),
    }

    # ---- 5. RQ2 with source-only read-out selection
    tv_path = args.run / "trivia_feats.npz"
    if tv_path.exists():
        from agentconf.io import read_jsonl
        from agentconf.runs import load_features

        meta = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))
        layers = meta["layers"]
        tv = np.load(tv_path)
        ty = np.array([r["correct"] for r in read_jsonl(args.run / "trivia.jsonl")], int)
        groups = np.arange(len(ty))
        sweep = []
        for pooling in ("mean", "last"):
            for j, l in enumerate(layers):
                a = auroc(oof_scores(tv[pooling][:, j, :], ty, groups, n_components=args.pca), ty)
                sweep.append({"pooling": pooling, "layer": l, "trivia_oof_auroc": a})
        best = max(sweep, key=lambda r: np.nan_to_num(r["trivia_oof_auroc"], nan=-1))
        jj = layers.index(best["layer"])
        feats = load_features(args.run, steps, best["pooling"])[:, jj, :]
        p = fit_predict(tv[best["pooling"]][:, jj, :], ty, feats[te], n_components=args.pca)
        out["rq2_source_only"] = {"selected": best, "agent_test_auroc": auroc(p, y[te]),
                                  "ece_raw": ece(p, y[te]), "sweep": sweep}

    # ---- 6. borderline Finish steps
    border = [{"qid": s["qid"], "t": s["t"], "split": [r["split"] for r in rows if r["qid"] == s["qid"]][0],
               "answer": s.get("arg", ""), "gold": next(e["gold"] for e in episodes if e["qid"] == s["qid"]),
               "f1": s.get("f1"), "label": s["correct"]}
              for s in steps if s["kind"] == "Finish" and s.get("f1") not in (None, "") and 0 < float(s["f1"]) < 1]
    write_csv(res / "review_finish_borderline.csv", border)
    fin_te = [b for b in border if b["split"] == "test"]
    out["finish_borderline"] = {"n_all_splits": len(border), "n_test": len(fin_te),
                                "test_labelled_correct": sum(b["label"] for b in fin_te),
                                "test_finish_total": int((te & (kind == "Finish")).sum())}

    (res / "review_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rq2_source_only"}, indent=1))
    if "rq2_source_only" in out:
        print(json.dumps({k: v for k, v in out["rq2_source_only"].items() if k != "sweep"}, indent=1))


if __name__ == "__main__":
    main()
