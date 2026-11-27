"""Main-results confirmation, protocol MC1 v1.1 (protocols/main_confirmation_v1.md, section 8). CPU.

Probes and baselines are fitted once on the legacy training split and scored once on the held-out final split.

    python scripts/07_preaction_features.py --run runs/qwen7b-final      # GPU, label-free; also llama8b-final
    python scripts/23_confirm_main.py --freeze                            # records code and input hashes
    python scripts/23_confirm_main.py                                     # checks, then scores once

Freezing refuses if a confirmation output already exists. Scoring first runs every label-free check (frozen hashes;
target model, revision, layers, manifest hash and split; the 600 final episodes complete with steps 1..n, finite
action and pre-action features of the right shape), and checks that the legacy read-outs re-selected on the
training split match the frozen ones. Only then does it score, writing results/<target>/main_*.json per model and
results/main_confirmation.json with the pre-registered flags (section 4) and framing (section 5).
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.checkpoints import run_lock, valid_features  # noqa: E402
from agentconf.io import read_jsonl, write_json  # noqa: E402
from agentconf.metrics import aurc, auroc, brier, ece, platt  # noqa: E402
from agentconf.probes import fit_predict, layer_sweep, oof_scores  # noqa: E402
from agentconf.runs import load_features, load_run  # noqa: E402
from agentconf.textbase import contexts, pin_encoder, resolve_encoder, select_and_fit  # noqa: E402
from agentconf.seqmodel import (KINDS, compare, design, fit, logit, posterior,  # noqa: E402
                                posterior_dataset, sigmoid, trajectory_confidences)

PROTOCOL = "main_confirmation_v1.1 (protocols/main_confirmation_v1.md, section 8)"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CODE = ["scripts/07_preaction_features.py", "scripts/06_robustness.py", "scripts/16_review_checks.py",
        "scripts/23_confirm_main.py"] + [f"agentconf/{m}.py" for m in (
            "checkpoints", "io", "metrics", "probes", "runs", "seqmodel", "textbase")]
DEFAULT_PAIRS = ["runs/qwen7b=runs/qwen7b-final", "runs/llama8b=runs/llama8b-final"]
# Read-outs reported in the paper (legacy training split). The official freeze must reproduce them.
REGISTERED = {
    "Qwen/Qwen2.5-7B-Instruct": {"action": ["mean", 20], "pre_step": 20, "pre_action": 20, "trivia": ["last", 20]},
    "meta-llama/Llama-3.1-8B-Instruct": {"action": ["last", 23], "pre_step": 18, "pre_action": 18,
                                         "trivia": ["last", 14]},
}
BASELINES = ["consistency", "verbalized", "lp_mean", "lp_min"]
BONFERRONI = 3  # primary claims P1-P3
PRE_POINTS = ["pre_step", "pre_action"]


def _module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)


def run_digest(run, extra=()):
    """One sha256 over a run's metadata, episodes, steps and every feature file (sorted by relative path)."""
    run = Path(run)
    paths = [run / n for n in ("meta.json", "episodes.jsonl", "steps.jsonl", *extra)]
    paths += sorted((run / "feats").glob("*.npz")) + sorted((run / "feats_pre").glob("*.npz"))
    digest = hashlib.sha256()
    for p in paths:
        digest.update(f"{p.relative_to(run).as_posix()}:{sha256(p)}\n".encode())
    return {"sha256": digest.hexdigest(), "n_files": len(paths)}


# ---------------------------------------------------------------- loading

def load_side(run, split):
    meta, episodes, steps = load_run(run)
    steps = [s for s in steps if s["split"] == split]
    qids = {s["qid"] for s in steps}
    episodes = [e for e in episodes if e["qid"] in qids]
    return meta, episodes, steps


def load_pre(run, steps):
    cache, out = {}, {p: [] for p in PRE_POINTS}
    for s in steps:
        if s["qid"] not in cache:
            with np.load(Path(run) / "feats_pre" / f"{s['qid']}.npz") as z:
                cache[s["qid"]] = {p: z[p] for p in PRE_POINTS}
        for p in PRE_POINTS:
            out[p].append(cache[s["qid"]][p][s["t"] - 1])
    return {p: np.stack(v).astype(np.float32) for p, v in out.items()}


def question_only(pre_step, steps):
    """The pre-step state of each episode's first step (prompt and question only), repeated for all its steps."""
    first = {}
    for i, s in enumerate(steps):
        if s["t"] == 1:
            first[s["qid"]] = i
    return np.stack([pre_step[first[s["qid"]]] for s in steps])


def select_readouts(train_run, steps, layers, pca):
    """The paper's selection rules on the legacy training split; returns read-outs and the features they need."""
    y = np.array([s["correct"] for s in steps], int)
    g = np.array([s["qid"] for s in steps])
    feats, sweeps = {}, []
    for pooling in ("mean", "last"):
        feats[pooling] = load_features(train_run, steps, pooling)
        sweeps += [(np.nan_to_num(a, nan=-1), pooling, l)
                   for l, a in layer_sweep(feats[pooling], y, g, layers, n_components=pca).items()]
    _, pooling, layer = max(sweeps, key=lambda r: r[0])
    out = {"action": [pooling, int(layer)]}
    pre = load_pre(train_run, steps)
    pre["question_only"] = question_only(pre["pre_step"], steps)
    for p in PRE_POINTS + ["question_only"]:
        sw = layer_sweep(pre[p], y, g, layers, n_components=pca)
        out[p] = int(max(sw, key=lambda l: np.nan_to_num(sw[l], nan=-1)))
    tv_path = Path(train_run) / "trivia_feats.npz"
    if tv_path.exists():
        ty = np.array([r["correct"] for r in read_jsonl(Path(train_run) / "trivia.jsonl")], int)
        with np.load(tv_path) as tv:
            sw = [(np.nan_to_num(auroc(oof_scores(tv[p][:, j, :], ty, np.arange(len(ty)), n_components=pca), ty),
                                 nan=-1), p, l) for p in ("mean", "last") for j, l in enumerate(layers)]
        _, p, l = max(sw, key=lambda r: r[0])
        out["trivia"] = [p, int(l)]
    return out, feats, pre


# ---------------------------------------------------------------- freeze and checks

def freeze(args, pairs):
    if args.frozen.exists() and not args.force:
        raise SystemExit(f"{rel(args.frozen)} exists; pass --force only before any confirmation scoring")
    if args.out.exists():
        raise SystemExit("A confirmation output already exists: freezing now would not be a pre-registration")
    out = {"protocol": PROTOCOL, "heldout_manifest": {"path": rel(args.manifest), "sha256": sha256(args.manifest)},
           "split": args.split, "output": rel(args.out),
           "scoring": {"pca": args.pca, "n_boot": args.n_boot, "draws": args.draws, "tune": args.tune,
                       "chains": args.chains, "few_shot": 50, "few_shot_reps": 20,
                       "text_encoder": pin_encoder(args.text_encoder)},
           "code": {c: sha256(ROOT / c) for c in CODE}, "models": {}}
    ids = json.loads(args.manifest.read_text(encoding="utf-8"))["splits"][args.split]
    for s, t in pairs:
        problems = check_target(ROOT / t, None, out, ids)
        if problems:
            raise SystemExit("cannot freeze; fix these first:\n  " + "\n  ".join(problems))
        meta, _, steps = load_side(ROOT / s, "train")
        readouts, _, _ = select_readouts(ROOT / s, steps, meta["layers"], args.pca)
        want = REGISTERED.get(meta["model"])
        if want is not None and {k: readouts.get(k) for k in want} != want:
            raise SystemExit(f"{s}: re-selected read-outs {readouts} differ from the paper's {want}")
        tmeta = json.loads((ROOT / t / "meta.json").read_text(encoding="utf-8"))
        out["models"][rel(ROOT / s)] = {
            "target_run": rel(ROOT / t), "model": meta["model"], "layers": meta["layers"],
            "max_steps": meta.get("max_steps", 6), "target_revision": tmeta.get("revision"), "readouts": readouts,
            "train_inputs": run_digest(ROOT / s, extra=[p for p in ("trivia.jsonl", "trivia_feats.npz")
                                                        if (ROOT / s / p).exists()]),
            "target_inputs": run_digest(ROOT / t)}
    args.frozen.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.frozen, out)
    print(f"froze {len(CODE)} code files, the manifest and {len(pairs)} models -> {rel(args.frozen)}")


def check_target(t, f, fz, ids):
    """Label-free checks of one held-out run; returns a list of problems."""
    t = Path(t)
    if not (t / "meta.json").exists():
        return [f"{rel(t)}: no agent run"]
    tm = json.loads((t / "meta.json").read_text(encoding="utf-8"))
    problems = []
    want = {"ids_sha256": fz["heldout_manifest"]["sha256"]}
    if f is not None:
        want.update(model=f["model"], layers=f["layers"], revision=f["target_revision"])
    for key, v in want.items():
        if tm.get(key) != v:
            problems.append(f"{rel(t)}: {key} is {tm.get(key)!r}, expected {v!r}")
    if fz["split"] not in (tm.get("splits") or "").split(","):
        problems.append(f"{rel(t)}: split {fz['split']} was not run")
    eps = {e["qid"]: e for e in read_jsonl(t / "episodes.jsonl")} if (t / "episodes.jsonl").exists() else {}
    missing, extra = set(ids) - set(eps), set(eps) - set(ids)
    if missing or extra:
        problems.append(f"{rel(t)}: {len(missing)} manifest questions without an episode, {len(extra)} extra")
    steps = defaultdict(dict)
    for s in read_jsonl(t / "steps.jsonl") if (t / "steps.jsonl").exists() else []:
        steps[s["qid"]][s["t"]] = s
    bad, bad_pre = [], []
    n_layers = len(tm.get("layers", []))
    for q in set(ids) & set(eps):
        n = eps[q]["n_steps"]
        if not (n > 0 and all(k in steps[q] and steps[q][k].get("split") == fz["split"] for k in range(1, n + 1))
                and valid_features(t / "feats" / f"{q}.npz", (n, n_layers))):
            bad.append(q)
        p = t / "feats_pre" / f"{q}.npz"
        try:
            with np.load(p) as z:
                ok = all(z[k].shape[:2] == (n, n_layers) and np.isfinite(z[k]).all() for k in PRE_POINTS)
        except Exception:  # noqa: BLE001 - any unreadable file is a failed check
            ok = False
        if not ok:
            bad_pre.append(q)
    if bad:
        problems.append(f"{rel(t)}: {len(bad)} episodes with missing steps, wrong split or bad features "
                        f"(e.g. {sorted(bad)[:3]})")
    if bad_pre:
        problems.append(f"{rel(t)}: {len(bad_pre)} episodes without valid pre-action features; run "
                        f"scripts/07_preaction_features.py --run {rel(t)} (e.g. {sorted(bad_pre)[:3]})")
    return problems


def check(args, fz, pairs):
    problems = [f"code changed: {c}" for c, h in fz["code"].items() if sha256(ROOT / c) != h]
    if sha256(args.manifest) != fz["heldout_manifest"]["sha256"]:
        problems.append("held-out manifest changed")
    ids = json.loads(args.manifest.read_text(encoding="utf-8"))["splits"][fz["split"]]
    for s, t in pairs:
        f = fz["models"][s]
        extra = [p for p in ("trivia.jsonl", "trivia_feats.npz") if (ROOT / s / p).exists()]
        if run_digest(ROOT / s, extra) != f["train_inputs"]:
            problems.append(f"{s}: legacy training inputs changed since the freeze")
        target_problems = check_target(ROOT / t, f, fz, ids)
        problems += target_problems
        if not target_problems and run_digest(ROOT / t) != f["target_inputs"]:
            problems.append(f"{t}: held-out inputs changed since the freeze")
    return problems


# ---------------------------------------------------------------- scoring helpers

def one_hot(values, levels):
    return np.array([[v == level for level in levels] for v in values], float)


class Boot:
    """Question bootstrap shared by every signal of one model."""

    def __init__(self, qids, n_boot, seed=0):
        self.qids = np.asarray(qids)
        ids = np.unique(self.qids)
        members = {g: np.flatnonzero(self.qids == g) for g in ids}
        rng = np.random.default_rng(seed)
        self.choices = [rng.choice(len(ids), len(ids)) for _ in range(n_boot)]
        self.idx = [np.concatenate([members[ids[c]] for c in ch]) for ch in self.choices]
        self.ids, self.members = ids, members

    def interval(self, stat, mask=None, level=0.95):
        """Point estimate on the original sample and a percentile interval; stat(index_array) -> float."""
        base = np.arange(len(self.qids)) if mask is None else np.flatnonzero(mask)
        point = stat(base)
        vals = []
        for i in self.idx:
            j = i if mask is None else i[mask[i]]
            v = stat(j)
            if np.isfinite(v):
                vals.append(v)
        a = (1 - level) / 2 * 100
        lo, hi = (np.percentile(vals, [a, 100 - a]) if vals else (np.nan, np.nan))
        return {"estimate": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "level": level,
                "n_boot_valid": len(vals)}


def pair_stats(q, y, score):
    """Per question: (wins, pairs) over its correct/wrong step pairs."""
    out = {}
    for g in np.unique(q):
        m = q == g
        pos, neg = score[m][y[m] == 1], score[m][y[m] == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            out[g] = (float(((d > 0) + 0.5 * (d == 0)).sum()), d.size)
    return out


def pooled_pairs_interval(boot, y, s1, s2, n_boot_level=0.95):
    """Within-question pooled-pair AUROC of s1, s2 and their difference, resampling the same questions."""
    st1, st2 = pair_stats(boot.qids, y, s1), pair_stats(boot.qids, y, s2)

    def pooled(st, ids):
        w = sum(st[g][0] for g in ids if g in st)
        n = sum(st[g][1] for g in ids if g in st)
        return w / n if n else np.nan

    def result(f):
        point = f(boot.ids)
        vals = [f(boot.ids[c]) for c in boot.choices]
        vals = [v for v in vals if np.isfinite(v)]
        a = (1 - n_boot_level) / 2 * 100
        lo, hi = np.percentile(vals, [a, 100 - a])
        return {"estimate": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "level": n_boot_level}

    return {"n_questions_with_both": len(st1), "n_pairs": int(sum(v[1] for v in st1.values())),
            "s1": result(lambda ids: pooled(st1, ids)), "s2": result(lambda ids: pooled(st2, ids)),
            "difference": result(lambda ids: pooled(st1, ids) - pooled(st2, ids))}


def nuts_diagnostics(idata):
    import arviz as az

    rhat = az.rhat(idata)
    ess = az.ess(idata)
    r = max(float(np.nanmax(rhat[v].values)) for v in rhat.data_vars)
    e = min(float(np.nanmin(ess[v].values)) for v in ess.data_vars)
    div = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else -1
    return {"rhat_max": r, "ess_bulk_min": e, "divergences": div}


# ---------------------------------------------------------------- scoring

def score_model(train_run, target_run, f, sc):
    text_features = _module("16_review_checks.py").text_features
    lenient_labels = _module("06_robustness.py").lenient_labels
    pca, n_boot = sc["pca"], sc["n_boot"]
    meta, episodes_tr, steps_tr = load_side(train_run, "train")
    _, episodes_te, steps_te = load_side(target_run, "final")
    layers = meta["layers"]
    readouts, feats_tr, pre_tr = select_readouts(train_run, steps_tr, layers, pca)
    if readouts != f["readouts"]:
        raise SystemExit(f"{rel(train_run)}: re-selected read-outs {readouts} differ from frozen {f['readouts']}")

    y_tr = np.array([s["correct"] for s in steps_tr], int)
    y = np.array([s["correct"] for s in steps_te], int)
    g_tr = np.array([s["qid"] for s in steps_tr])
    q = np.array([s["qid"] for s in steps_te])
    t = np.array([s["t"] for s in steps_te])
    kind = np.array([s["kind"] for s in steps_te])
    boot = Boot(q, n_boot, seed=0)

    # ---- probes, fitted on legacy train
    pooling, layer = readouts["action"]
    j = layers.index(layer)
    X_tr = feats_tr[pooling][:, j, :]
    X_te = load_features(target_run, steps_te, pooling)[:, j, :]
    sig = {"probe_action": fit_predict(X_tr, y_tr, X_te, n_components=pca)}
    probe_oof_tr = oof_scores(X_tr, y_tr, g_tr, n_components=pca)
    pre_te = load_pre(target_run, steps_te)
    for p in PRE_POINTS:
        jj = layers.index(readouts[p])
        sig[f"probe_{p}"] = fit_predict(pre_tr[p][:, jj, :], y_tr, pre_te[p][:, jj, :], n_components=pca)
    jq = layers.index(readouts["question_only"])
    Q_tr = pre_tr["question_only"][:, jq, :]
    Q_te = question_only(pre_te["pre_step"], steps_te)[:, jq, :]
    sig["question_only"] = fit_predict(Q_tr, y_tr, Q_te, n_components=pca)
    q_oof_tr = oof_scores(Q_tr, y_tr, g_tr, n_components=pca)

    # ---- output signals (raw for ranking, Platt on legacy train for probabilities)
    prob = {k: v for k, v in sig.items()}
    for b in BASELINES:
        raw_tr = np.array([s[b] if s.get(b) is not None else np.nan for s in steps_tr], float)
        raw_te = np.array([s[b] if s.get(b) is not None else np.nan for s in steps_te], float)
        fill = np.nanmedian(raw_tr) if np.isfinite(raw_tr).any() else 0.0
        raw_tr, raw_te = np.nan_to_num(raw_tr, nan=fill), np.nan_to_num(raw_te, nan=fill)
        sig[b], prob[b] = raw_te, platt(raw_tr, y_tr, raw_te)

    # ---- visible text + history, and position baselines
    tf_tr, tf_te = text_features(steps_tr), text_features(steps_te)
    names = list(next(iter(tf_tr.values())))
    T_tr = np.array([[tf_tr[(s["qid"], s["t"])][k] for k in names] for s in steps_tr], float)
    T_te = np.array([[tf_te[(s["qid"], s["t"])][k] for k in names] for s in steps_te], float)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(T_tr, y_tr)
    sig["visible_text"] = prob["visible_text"] = lr.predict_proba(T_te)[:, 1]  # v1 baseline, kept as an estimate
    ctx_tr = contexts(steps_tr, {e["qid"]: e["question"] for e in episodes_tr})
    ctx_te = contexts(steps_te, {e["qid"]: e["question"] for e in episodes_te})
    text_scores, text_record = select_and_fit(ctx_tr, T_tr, y_tr, g_tr, ctx_te, T_te,
                                              resolve_encoder(sc["text_encoder"]))
    for k, v in text_scores.items():
        sig[f"text_{k}"] = prob[f"text_{k}"] = v
    sig["text_best"] = prob["text_best"] = text_scores[text_record["selected"]]
    levels_t = list(range(1, int(f.get("max_steps", 6)) + 1))
    kinds_sorted = sorted(KINDS)
    t_tr = np.array([s["t"] for s in steps_tr])
    k_tr = np.array([s["kind"] for s in steps_tr])
    P_tr, P_te = one_hot(t_tr, levels_t), one_hot(t, levels_t)
    PK_tr = np.hstack([P_tr, one_hot(k_tr, kinds_sorted)])
    PK_te = np.hstack([P_te, one_hot(kind, kinds_sorted)])
    for name, A, B in (("step_index", P_tr, P_te), ("step_index_type", PK_tr, PK_te),
                       ("question_only_step_index", np.hstack([P_tr, logit(q_oof_tr)[:, None]]),
                        np.hstack([P_te, logit(sig["question_only"])[:, None]]))):
        sig[name] = prob[name] = LogisticRegression(C=1e4, max_iter=2000).fit(A, y_tr).predict_proba(B)[:, 1]

    A = lambda s: (lambda i: auroc(sig[s][i], y[i]))  # noqa: E731
    D = lambda a, b: (lambda i: auroc(sig[a][i], y[i]) - auroc(sig[b][i], y[i]))  # noqa: E731
    search, finish = kind == "Search", kind == "Finish"
    out = {"n_steps": int(len(y)), "n_questions": int(len(boot.ids)), "step_accuracy": float(y.mean()),
           "readouts": readouts, "text_baselines": text_record, "signals": {}}
    for s in sig:
        out["signals"][s] = {"auroc": boot.interval(A(s)), "ece": ece(prob[s], y), "brier": brier(prob[s], y),
                             "auroc_search": auroc(sig[s][search], y[search]),
                             "auroc_finish": auroc(sig[s][finish], y[finish])}

    lvl = 1 - 0.05 / BONFERRONI
    out["primary"] = {
        "P1_probe_minus_consistency": boot.interval(D("probe_action", "consistency"), level=lvl),
        "P2_probe_minus_text_best": boot.interval(D("probe_action", "text_best"), level=lvl),
        "P3_prestep_minus_step_index": boot.interval(D("probe_pre_step", "step_index"), level=lvl),
    }
    within = pooled_pairs_interval(boot, y, sig["probe_action"], sig["consistency"])
    out["secondary"] = {
        "S1_search_probe_minus_consistency": boot.interval(D("probe_action", "consistency"), mask=search),
        "S2_within_question_probe_minus_consistency": within["difference"],
        "S3_probe_minus_step_index_type": boot.interval(D("probe_action", "step_index_type")),
        "S6_prestep_minus_question_only_step_index": boot.interval(D("probe_pre_step", "question_only_step_index")),
    }
    est = {"primary_at_95": {k: boot.interval(D(*pair)) for k, pair in (
        ("P1", ("probe_action", "consistency")), ("P2", ("probe_action", "text_best")),
        ("P3", ("probe_pre_step", "step_index")))}}
    est["within_question"] = within
    est["finish_probe_minus_consistency"] = boot.interval(D("probe_action", "consistency"), mask=finish)
    for p in PRE_POINTS:
        est[f"{p}_minus_consistency"] = boot.interval(D(f"probe_{p}", "consistency"))
        est[f"{p}_minus_probe_action"] = boot.interval(D(f"probe_{p}", "probe_action"))
    for b in ("verbalized", "lp_mean", "lp_min", "visible_text", "text_handcrafted", "text_tfidf",
              "text_embedding"):
        est[f"probe_minus_{b}"] = boot.interval(D("probe_action", b))
    est["question_only_minus_step_index"] = boot.interval(D("question_only", "step_index"))
    est["by_step_index"] = {}
    for k in levels_t:
        m = t == k
        if 0 < y[m].sum() < m.sum():
            est["by_step_index"][k] = {"n": int(m.sum()), "accuracy": float(y[m].mean()),
                                       **{s: auroc(sig[s][m], y[m]) for s in ("probe_action", "probe_pre_step",
                                                                               "consistency")}}
    yl = lenient_labels(steps_te)
    est["lenient"] = {"labels_changed": int((yl != y).sum()),
                      "probe": auroc(sig["probe_action"], yl), "consistency": auroc(sig["consistency"], yl),
                      "probe_minus_consistency": boot.interval(
                          lambda i: auroc(sig["probe_action"][i], yl[i]) - auroc(sig["consistency"][i], yl[i]))}

    # ---- RQ2: TriviaQA transfer
    tv_path = Path(train_run) / "trivia_feats.npz"
    if tv_path.exists():
        ty = np.array([r["correct"] for r in read_jsonl(Path(train_run) / "trivia.jsonl")], int)
        rq2 = {}
        with np.load(tv_path) as tv:
            for name, (p, l) in (("source_only_readout", readouts["trivia"]),
                                 ("agent_label_readout", readouts["action"])):
                jj = layers.index(l)
                F_te = load_features(target_run, steps_te, p)[:, jj, :]
                F_tr = feats_tr[p][:, jj, :]
                z_te = fit_predict(tv[p][:, jj, :], ty, F_te, n_components=pca)
                z_tr = fit_predict(tv[p][:, jj, :], ty, F_tr, n_components=pca)
                r = {"readout": [p, l], "auroc": boot.interval(lambda i, z=z_te: auroc(z[i], y[i])),
                     "ece_raw": ece(z_te, y)}
                rng = np.random.default_rng(0)
                recal, small = [], []
                for _ in range(sc["few_shot_reps"]):
                    pick = rng.choice(len(y_tr), size=min(sc["few_shot"], len(y_tr)), replace=False)
                    if len(set(y_tr[pick])) < 2:
                        continue
                    pr = platt(z_tr[pick], y_tr[pick], z_te)
                    recal.append((ece(pr, y), brier(pr, y)))
                    ps = fit_predict(F_tr[pick], y_tr[pick], F_te, n_components=min(pca, len(pick) - 1))
                    small.append(auroc(ps, y))
                r["platt_few_shot"] = {"ece_mean": float(np.mean([a for a, _ in recal])),
                                       "ece_sd": float(np.std([a for a, _ in recal])),
                                       "brier_mean": float(np.mean([b for _, b in recal])), "reps": len(recal)}
                r["in_domain_few_shot_auroc"] = {"mean": float(np.mean(small)), "sd": float(np.std(small))}
                rq2[name] = r
        est["rq2"] = rq2

    # ---- RQ3
    out["rq3"] = rq3(steps_tr, probe_oof_tr, steps_te, sig["probe_action"], episodes_te, sc)
    out["secondary"]["S5_g_final_q95"] = out["rq3"]["final_fit"]["g"]
    out["estimates"] = est
    return out


def rq3(steps_tr, s_tr, steps_te, s_te, episodes_te, sc):
    kw = dict(draws=sc["draws"], tune=sc["tune"], chains=sc["chains"])
    rows = lambda st, s: [{"qid": x["qid"], "t": x["t"], "kind": x["kind"], "correct": int(x["correct"]),  # noqa
                           "s": float(logit(v))} for x, v in zip(st, s)]
    tr_rows, te_rows = rows(steps_tr, s_tr), rows(steps_te, s_te)
    id0 = fit(design(tr_rows), coupled=False, seed=0, **kw)
    id1 = fit(design(tr_rows), coupled=True, seed=1, **kw)
    p0, p1 = posterior(id0), posterior(id1)
    out = {"train_fit": {"M0": nuts_diagnostics(id0), "M1": nuts_diagnostics(id1)}}

    by_ep = defaultdict(list)
    for r in te_rows:
        by_ep[r["qid"]].append(r)
    ep = {e["qid"]: e for e in episodes_te}
    recs = []
    for qid in sorted(by_ep):
        st = sorted(by_ep[qid], key=lambda r: r["t"])
        c = trajectory_confidences(st, p0, p1)
        a0, b0, _ = p0
        prod, run0 = np.ones(a0.shape[0]), []
        for s in st:
            k = KINDS.index(s["kind"])
            prod = prod * sigmoid(a0[:, k] + b0[:, k] * s["s"])
            run0.append(float(prod.mean()))
        recs.append({"qid": qid, "finished": bool(ep[qid]["finished"]), "success": int(ep[qid]["success"]),
                     "M0_last": c["M0_last"], "M0_product": c["M0_product"], "M0_min": c["M0_min"],
                     "M1_forward": c["M1_forward"], "handoff_M1": min(c["running"]), "handoff_M0": min(run0)})

    def block(sel, seed=1):
        """Unfinished episodes reached the step cap without an answer, which the system knows at that point:
        every method gives them confidence 0 (handed off at the cap), never a probability of another event."""
        r = [x for x in recs if sel(x)]
        succ = np.array([x["success"] for x in r])
        done = np.array([x["finished"] for x in r])
        v = {m: np.where(done, np.array([x[m] for x in r]), 0.0)
             for m in ("M0_last", "M0_product", "M0_min", "M1_forward", "handoff_M0", "handoff_M1")}
        b = Boot([x["qid"] for x in r], sc["n_boot"], seed=seed)
        res = {"n_episodes": len(r), "success_rate": float(succ.mean()),
               "methods": {m: {"auroc": auroc(v[m], succ), "ece": ece(v[m], succ), "brier": brier(v[m], succ),
                               "mean_conf": float(v[m].mean())}
                           for m in ("M0_last", "M0_product", "M0_min", "M1_forward")}}
        # Fair comparison (same event): forward vs last. The product targets "every step correct" (descriptive).
        res["ece_product_minus_forward"] = b.interval(lambda i: ece(v["M0_product"][i], succ[i])
                                                      - ece(v["M1_forward"][i], succ[i]))
        res["ece_last_minus_forward"] = b.interval(lambda i: ece(v["M0_last"][i], succ[i])
                                                   - ece(v["M1_forward"][i], succ[i]))
        res["auroc_forward_minus_last"] = b.interval(lambda i: auroc(v["M1_forward"][i], succ[i])
                                                     - auroc(v["M0_last"][i], succ[i]))
        res["brier_forward_minus_last"] = b.interval(lambda i: brier(v["M1_forward"][i], succ[i])
                                                     - brier(v["M0_last"][i], succ[i]))
        res["aurc_handoff_M0_minus_M1"] = b.interval(lambda i: aurc(v["handoff_M0"][i], succ[i])
                                                     - aurc(v["handoff_M1"][i], succ[i]))
        return res

    out["finished"] = block(lambda x: x["finished"])
    out["all_episodes"] = block(lambda x: True)
    out["n_unfinished"] = sum(not x["finished"] for x in recs)

    f0 = fit(design(te_rows), coupled=False, seed=0, **kw)
    f1 = fit(design(te_rows), coupled=True, seed=1, **kw)
    g = posterior_dataset(f1)["g"].values.ravel()
    out["final_fit"] = {"loo": compare(f0, f1), "M0": nuts_diagnostics(f0), "M1": nuts_diagnostics(f1),
                        "g": {"mean": float(g.mean()), "q05": float(np.quantile(g, 0.05)),
                              "q95": float(np.quantile(g, 0.95))}}
    out["episodes"] = recs
    return out


# ---------------------------------------------------------------- decisions

def dev_check(args, pairs):
    """Text baselines fitted on legacy train and scored on the legacy (development) test split. Reads no held-out
    data; meant to catch implementation problems on real trajectories before the freeze."""
    text_features = _module("16_review_checks.py").text_features
    encoder = resolve_encoder(pin_encoder(args.text_encoder))
    for s, _ in pairs:
        _, eps, steps = load_run(ROOT / s)
        q = {e["qid"]: e["question"] for e in eps}
        tr = [x for x in steps if x["split"] == "train"]
        te = [x for x in steps if x["split"] == "test"]
        tf_tr, tf_te = text_features(tr), text_features(te)
        names = list(next(iter(tf_tr.values())))
        T_tr = np.array([[tf_tr[(x["qid"], x["t"])][k] for k in names] for x in tr], float)
        T_te = np.array([[tf_te[(x["qid"], x["t"])][k] for k in names] for x in te], float)
        y_tr = np.array([x["correct"] for x in tr], int)
        y_te = np.array([x["correct"] for x in te], int)
        scores, rec = select_and_fit(contexts(tr, q), T_tr, y_tr, np.array([x["qid"] for x in tr]),
                                     contexts(te, q), T_te, encoder)
        print(s, "selected:", rec["selected"], {k: round(v["oof_auroc_train"], 3) for k, v in rec["candidates"].items()},
              "| development test AUROC:", {k: round(auroc(v, y_te), 3) for k, v in scores.items()})


def flags_for(o):
    above = lambda r: r["ci_lo"] > 0  # noqa: E731
    sec = o["secondary"]
    return {"P1": above(o["primary"]["P1_probe_minus_consistency"]),
            "P2": above(o["primary"]["P2_probe_minus_text_best"]),
            "P3": above(o["primary"]["P3_prestep_minus_step_index"]),
            "S1": above(sec["S1_search_probe_minus_consistency"]),
            "S2": above(sec["S2_within_question_probe_minus_consistency"]),
            "S3": above(sec["S3_probe_minus_step_index_type"]),
            "S5": sec["S5_g_final_q95"]["q95"] < 0,
            "S6": above(sec["S6_prestep_minus_question_only_step_index"])}


def framing(per_model_flags):
    """Section 5: every condition must hold in every model."""
    allm = {k: all(f[k] for f in per_model_flags.values()) for k in next(iter(per_model_flags.values()))}
    text = []
    if not allm["P1"]:
        text.append("P1 not confirmed: the main RQ1 result does not replicate; report as negative, shift weight "
                    "to RQ2/RQ3")
    else:
        text.append("P1 confirmed: report final-split numbers in the main table")
        text.append("P2 confirmed: the probe beats the strongest pre-specified text baseline" if allm["P2"] else
                    "P2 not confirmed: claim only that the probe beats output-based signals; report the text "
                    "baselines as matching it")
    text.append("P3 confirmed: keep the pre-step detection claim" if allm["P3"] else
                "P3 not confirmed: drop the pre-step detection claim; pre-step results move to an appendix")
    if allm["P3"]:
        text.append("S6 confirmed: the pre-step signal goes beyond question difficulty and position" if allm["S6"]
                    else "S6 not confirmed: describe the pre-step signal as largely question difficulty and position")
    text.append("S5 confirmed: keep the error-association description" if allm["S5"] else
                "S5 not confirmed: say the association was observed on the legacy split only")
    return allm, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", help="train_run=target_run per model (freeze only)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/heldout_v1/heldout_manifest.json")
    ap.add_argument("--split", default="final")
    ap.add_argument("--frozen", type=Path, default=ROOT / "manifests/main_confirmation/v1_frozen_inputs.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/main_confirmation.json")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dev-check", action="store_true", help="text baselines on the development test split only")
    ap.add_argument("--text-encoder", default=DEFAULT_ENCODER, help=argparse.SUPPRESS)  # synthetic freezes: hash:<d>
    for k in ("pca", "n-boot", "draws", "tune", "chains"):  # synthetic freezes only
        ap.add_argument(f"--{k}", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.frozen.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(args.frozen.parent):
        return run(args)


def run(args):
    supplied = any(getattr(args, k) is not None for k in ("pca", "n_boot", "draws", "tune", "chains"))
    official = args.frozen.resolve() == (ROOT / "manifests/main_confirmation/v1_frozen_inputs.json").resolve()
    if args.dev_check:
        return dev_check(args, [tuple(p.split("=", 1)) for p in args.pairs or DEFAULT_PAIRS])
    if args.freeze:
        if official and (args.pairs or supplied or args.split != "final" or args.text_encoder != DEFAULT_ENCODER):
            raise SystemExit("Use the registered models, final split and scoring settings for the official freeze")
        for k, v in (("pca", 64), ("n_boot", 2000), ("draws", 1000), ("tune", 1000), ("chains", 4)):
            setattr(args, k, v if getattr(args, k) is None else getattr(args, k))
        return freeze(args, [tuple(p.split("=", 1)) for p in args.pairs or DEFAULT_PAIRS])
    if args.pairs or supplied:
        raise SystemExit("Models and scoring settings were fixed at freeze time")
    if not args.frozen.exists():
        raise SystemExit(f"{rel(args.frozen)} does not exist: run with --freeze first")
    fz = json.loads(args.frozen.read_text(encoding="utf-8"))
    if args.out.resolve() != (ROOT / fz["output"]).resolve():
        raise SystemExit("Use the frozen output path")
    frozen_hash = sha256(args.frozen)
    if args.out.exists():
        if json.loads(args.out.read_text(encoding="utf-8")).get("frozen_sha256") != frozen_hash:
            raise SystemExit("The output belongs to another freeze")
        print(f"Confirmation already completed: {rel(args.out)}")
        return
    pairs = [(s, f["target_run"]) for s, f in fz["models"].items()]
    problems = check(args, fz, pairs)
    if problems:
        raise SystemExit("not scoring; fix these first:\n  " + "\n  ".join(problems))

    per_model = {}
    for s, t in pairs:
        print(f"scoring {t} with probes from {s}", flush=True)
        per_model[s] = score_model(ROOT / s, ROOT / t, fz["models"][s], fz["scoring"])
    flags = {s: flags_for(o) for s, o in per_model.items()}
    joint, text = framing(flags)
    for (s, t), o in zip(pairs, per_model.values()):
        dest = ROOT / t / "results"
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / "main_confirmation_model.json", o)
    summary = {"protocol": PROTOCOL, "split": fz["split"], "frozen_sha256": frozen_hash,
               "models": list(per_model), "joint": joint, "framing": text, "flags": flags,
               "primary": {s: o["primary"] for s, o in per_model.items()},
               "secondary": {s: o["secondary"] for s, o in per_model.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, summary)
    for s in per_model:
        print(s, {k: round(v["estimate"], 3) for k, v in per_model[s]["primary"].items()}, flags[s])
    print("\n".join(text))


if __name__ == "__main__":
    main()
