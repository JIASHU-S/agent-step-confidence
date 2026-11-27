"""Manual audit of the automatic step labels (protocols/label_audit_v1.md). CPU, no model.

Sample once from the development runs (both models), write a blinded annotation sheet and a separate key:
    python scripts/24_label_audit.py sample --runs runs/qwen7b runs/llama8b --out audit/label_audit_v1
Each annotator fills a copy of annotate_steps.csv and annotate_episodes.csv (codes in
protocols/label_audit_guidelines.md). Then:
    python scripts/24_label_audit.py score --out audit/label_audit_v1 --annotator A=<dir> [--annotator B=<dir>]

The sheets show the question, gold answer, the history the agent saw, the step and the returned paragraph. They do
not show the model, the automatic label, the F1 score or which paragraphs are supporting. Items are ordered by a
salted hash so strata and models are interleaved. Scoring reports, per annotator, agreement with the automatic
label (strict and lenient human codes), per stratum and population-weighted, and Cohen's kappa between annotators.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.io import write_json  # noqa: E402
from agentconf.runs import load_run  # noqa: E402

SALT = "label-audit-v1:"
# Steps per model per stratum (protocol section 2). Strata are disjoint and cover every recorded step.
QUOTA = {"search_correct": 25, "search_distractor": 25, "search_repeat": 15, "finish_correct_exact": 7,
         "finish_correct_partial": 8, "finish_wrong_partial": 8, "finish_wrong_zero": 7, "invalid": 5}
EPISODES_PER_MODEL = 20
STEP_CODES = {"Search": {"S-USEFUL": 1, "S-PARTIAL": None, "S-UNHELPFUL": 0, "S-REPEAT": 0, "UNCLEAR": None},
              "Finish": {"F-CORRECT": 1, "F-PARTIAL": None, "F-WRONG": 0, "F-GOLD-ISSUE": None, "UNCLEAR": None},
              "invalid": {"I-INVALID": 0, "I-VALID": None, "UNCLEAR": None}}
LENIENT = {"S-PARTIAL": 1, "F-PARTIAL": 1, "I-VALID": None}  # lenient human reading; others as strict
EPISODE_CODES = {"E-CORRECT": 1, "E-PARTIAL": None, "E-WRONG": 0, "E-GOLD-ISSUE": None, "UNCLEAR": None}
STEP_FIELDS = ["item_id", "episode_item", "question", "gold_answer", "history", "step_text", "action", "argument",
               "returned_title", "returned_text", "code", "notes"]
EPISODE_FIELDS = ["episode_id", "question", "gold_answer", "transcript", "final_answer", "code", "notes"]


def h(*parts):
    return hashlib.sha256((SALT + ":".join(map(str, parts))).encode()).hexdigest()


def stratum(step, seen_titles):
    if step["kind"] == "invalid":
        return "invalid"
    if step["kind"] == "Search":
        if step["correct"]:
            return "search_correct"
        return "search_repeat" if step.get("title") in seen_titles else "search_distractor"
    f1 = float(step.get("f1") or 0.0)
    if step["correct"]:
        return "finish_correct_exact" if f1 >= 1.0 else "finish_correct_partial"
    return "finish_wrong_partial" if f1 > 0 else "finish_wrong_zero"


def annotate_strata(steps):
    """Adds `stratum` and `history` (what the agent had seen before this step) to every step, in place."""
    by_q = defaultdict(list)
    for s in steps:
        by_q[s["qid"]].append(s)
    for st in by_q.values():
        st.sort(key=lambda s: s["t"])
        seen, hist = set(), []
        for s in st:
            s["stratum"] = stratum(s, seen)
            s["history"] = "\n".join(hist) if hist else "(first step)"
            hist.append(f"[{s['t']}] {s['kind']}[{s.get('arg', '')}] -> {s.get('title') or '-'}")
            if s["kind"] == "Search" and s.get("title"):
                seen.add(s["title"])
    return by_q


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so spreadsheet apps read UTF-8
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def step_row(item_id, ep, s, episode_item=""):
    return {"item_id": item_id, "episode_item": episode_item, "question": ep["question"],
            "gold_answer": ep["gold"], "history": s["history"], "step_text": s.get("text", ""),
            "action": s["kind"], "argument": s.get("arg", ""),
            "returned_title": (s.get("title") or "") if s["kind"] == "Search" else "",
            "returned_text": s.get("obs", "") if s["kind"] == "Search" else "", "code": "", "notes": ""}


def sample(args):
    if (args.out / "key.json").exists() and not args.force:
        raise SystemExit(f"{args.out}/key.json exists: the audit sample is drawn once")
    steps_out, eps_out, key_steps, key_eps, population = [], [], [], [], {}
    for k, run in enumerate(args.runs):
        meta, episodes, steps = load_run(run)
        model = meta["model"]
        by_q = annotate_strata(steps)
        ep_by = {e["qid"]: e for e in episodes}
        # 1. complete episodes, stratified by success and by length (<=2 steps vs longer)
        groups = defaultdict(list)
        for e in episodes:
            groups[(int(e["success"]), e["n_steps"] <= 2)].append(e)
        chosen = []
        per = EPISODES_PER_MODEL // max(len(groups), 1)
        for g in sorted(groups):
            chosen += sorted(groups[g], key=lambda e: h("ep", model, e["qid"]))[:per]
        rest = sorted([e for e in episodes if e not in chosen], key=lambda e: h("ep", model, e["qid"]))
        chosen += rest[:EPISODES_PER_MODEL - len(chosen)]
        ep_qids = {e["qid"] for e in chosen}
        for e in chosen:
            eid = h("episode", model, e["qid"])[:10]
            st = by_q[e["qid"]]
            transcript = "\n\n".join(f"{s.get('text', '')}\nObservation {s['t']}: {s.get('obs', '')}" for s in st)
            last = st[-1] if st else None
            eps_out.append({"episode_id": eid, "question": e["question"], "gold_answer": e["gold"],
                            "transcript": transcript,
                            "final_answer": last.get("arg", "") if last and last["kind"] == "Finish" else "(none)",
                            "code": "", "notes": ""})
            key_eps.append({"episode_id": eid, "model": model, "qid": e["qid"], "success": int(e["success"]),
                            "finished": bool(e.get("finished", True)), "n_steps": e["n_steps"]})
            for s in st:
                iid = h("step", model, s["qid"], s["t"])[:10]
                steps_out.append(step_row(iid, e, s, episode_item=eid))
                key_steps.append({"item_id": iid, "model": model, "qid": s["qid"], "t": s["t"], "kind": s["kind"],
                                  "auto_label": int(s["correct"]), "stratum": s["stratum"], "f1": s.get("f1"),
                                  "source": "episode"})
        # 2. stratified single steps from the other questions
        pool = defaultdict(list)
        for s in steps:
            if s["qid"] not in ep_qids:
                pool[s["stratum"]].append(s)
        population[model] = {st: len(v) for st, v in pool.items()}
        for st, quota in QUOTA.items():
            # Equal-probability sampling without replacement within the stratum (salted-hash order), so every step
            # of the stratum has inclusion probability n_sampled / n_population.
            taken = sorted(pool.get(st, []), key=lambda s: h("pick", model, s["qid"], s["t"]))[:quota]
            for s in taken:
                iid = h("step", model, s["qid"], s["t"])[:10]
                steps_out.append(step_row(iid, ep_by[s["qid"]], s))
                key_steps.append({"item_id": iid, "model": model, "qid": s["qid"], "t": s["t"], "kind": s["kind"],
                                  "auto_label": int(s["correct"]), "stratum": st, "f1": s.get("f1"),
                                  "source": "stratified", "n_population": len(pool.get(st, [])),
                                  "n_sampled": len(taken)})
    steps_out.sort(key=lambda r: h("order", r["item_id"]))
    eps_out.sort(key=lambda r: h("order", r["episode_id"]))
    write_csv(args.out / "annotate_steps.csv", steps_out, STEP_FIELDS)
    write_csv(args.out / "annotate_episodes.csv", eps_out, EPISODE_FIELDS)
    write_json(args.out / "key.json", {"protocol": "label_audit_v1", "runs": [str(r) for r in args.runs],
                                       "quota": QUOTA, "population": population,
                                       "steps": key_steps, "episodes": key_eps})
    print(f"{len(steps_out)} step items ({sum(r['source'] == 'episode' for r in key_steps)} from "
          f"{len(eps_out)} complete episodes) -> {args.out}")


# ---------------------------------------------------------------- scoring

def kappa(a, b):
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / len(a) ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - r), float(c + r)]


def load_annotations(directory, key):
    d = Path(directory)
    steps = {r["item_id"]: r["code"].strip().upper() for r in read_csv(d / "annotate_steps.csv")}
    eps = {r["episode_id"]: r["code"].strip().upper() for r in read_csv(d / "annotate_episodes.csv")}
    kind = {r["item_id"]: r["kind"] for r in key["steps"]}
    bad = [i for i, c in steps.items() if c and c not in STEP_CODES[kind[i]]]
    bad += [i for i, c in eps.items() if c and c not in EPISODE_CODES]
    if bad:
        raise SystemExit(f"{directory}: unknown codes for items {bad[:5]} (see protocols/label_audit_guidelines.md)")
    return steps, eps


def human_label(kind, code, lenient=False):
    if not code:
        return None
    if lenient and code in LENIENT:
        return LENIENT[code]
    return STEP_CODES[kind][code]


def agreement(key, codes, lenient):
    rows = []
    for r in key["steps"]:
        y = human_label(r["kind"], codes.get(r["item_id"], ""), lenient)
        if y is not None:
            rows.append((r, y))
    out = {"n_scored": len(rows), "by_stratum": {}}
    for st in QUOTA:
        rs = [(r, y) for r, y in rows if r["stratum"] == st]
        n_all = sum(1 for r in key["steps"] if r["stratum"] == st and codes.get(r["item_id"]))
        k = sum(int(r["auto_label"] == y) for r, y in rs)
        out["by_stratum"][st] = {"n": len(rs), "agree": k, "rate": k / len(rs) if rs else float("nan"),
                                 "wilson95": wilson(k, len(rs)), "n_coded": n_all,
                                 "n_excluded": n_all - len(rs),
                                 "excluded_share": (n_all - len(rs)) / n_all if n_all else float("nan")}
    # Population-weighted agreement from the stratified part, per model: stratum weight n_population, estimand =
    # agreement among steps whose human code is decisive (excluded codes are reported separately above). 95% CI from
    # a bootstrap that resamples questions within each stratum.
    out["weighted_accuracy"], out["weighted_accuracy_ci95"] = {}, {}
    rng = np.random.default_rng(0)
    for model, pop in key["population"].items():
        cells = {st: [(r["qid"], int(r["auto_label"] == y)) for r, y in rows
                      if r["stratum"] == st and r["model"] == model and r["source"] == "stratified"]
                 for st in pop}
        cells = {st: c for st, c in cells.items() if c}

        def estimate(cs):
            den = sum(pop[st] for st in cs)
            return sum(pop[st] * np.mean([a for _, a in c]) for st, c in cs.items()) / den if den else np.nan

        out["weighted_accuracy"][model] = float(estimate(cells))
        reps = []
        for _ in range(1000):
            res = {}
            for st, c in cells.items():
                by_q = defaultdict(list)
                for q, a in c:
                    by_q[q].append((q, a))
                qs = list(by_q)
                res[st] = [x for i in rng.integers(0, len(qs), len(qs)) for x in by_q[qs[i]]]
            reps.append(estimate(res))
        out["weighted_accuracy_ci95"][model] = ([float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]
                                                if cells else [float("nan")] * 2)
    conf = Counter((r["kind"], r["auto_label"], y) for r, y in rows)
    out["confusion"] = [{"kind": k, "auto": a, "human": y, "n": n} for (k, a, y), n in sorted(conf.items())]
    return out


def score(args):
    key = json.loads((args.out / "key.json").read_text(encoding="utf-8"))
    ann = {}
    for spec in args.annotator:
        name, d = spec.split("=", 1)
        ann[name] = load_annotations(d, key)
    res = {"protocol": "label_audit_v1", "annotators": {}}
    for name, (steps, eps) in ann.items():
        ep_rows = [(r, EPISODE_CODES[eps[r["episode_id"]]]) for r in key["episodes"] if eps.get(r["episode_id"])]
        ep_rows = [(r, y) for r, y in ep_rows if y is not None]
        res["annotators"][name] = {
            "missing_step_codes": sum(1 for r in key["steps"] if not steps.get(r["item_id"])),
            "code_counts": dict(Counter(c for c in steps.values() if c)),
            "strict": agreement(key, steps, lenient=False), "lenient": agreement(key, steps, lenient=True),
            "episode_success_agreement": {"n": len(ep_rows),
                                          "agree": sum(int(r["success"] == y) for r, y in ep_rows)}}
    names = list(ann)
    if len(names) >= 2:
        a, b = ann[names[0]][0], ann[names[1]][0]
        both = [i for i in a if a[i] and b.get(i)]
        res["inter_annotator"] = {"pair": names[:2], "n": len(both),
                                  "kappa_codes": kappa([a[i] for i in both], [b[i] for i in both]),
                                  "raw_agreement": float(np.mean([a[i] == b[i] for i in both])) if both else None,
                                  "disagreements": [i for i in both if a[i] != b[i]]}
    write_json(args.out / "audit_results.json", res)
    for name, r in res["annotators"].items():
        print(name, "strict weighted accuracy:", {m: round(v, 3) for m, v in r["strict"]["weighted_accuracy"].items()})
    if "inter_annotator" in res:
        print("kappa:", round(res["inter_annotator"]["kappa_codes"], 3))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--runs", nargs="+", type=Path, required=True)
    s.add_argument("--out", type=Path, default=ROOT / "audit/label_audit_v1")
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("score")
    c.add_argument("--out", type=Path, default=ROOT / "audit/label_audit_v1")
    c.add_argument("--annotator", action="append", required=True, help="NAME=directory with filled sheets")
    args = ap.parse_args()
    (sample if args.cmd == "sample" else score)(args)


if __name__ == "__main__":
    main()
