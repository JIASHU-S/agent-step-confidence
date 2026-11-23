"""Select the held-out HotpotQA questions once and save them as a manifest (agentconf/heldout.py has the rule).

    python scripts/21_select_heldout.py

Reads only IDs and question texts: the pinned HotpotQA validation file (pool), the legacy 900 IDs from the runs'
meta.json (must agree across runs), and the reference questions (legacy targets, all TriviaQA validation questions,
all 2Wiki dev questions). Writes manifests/heldout_v1/heldout_manifest.json; refuses to overwrite it.

    python scripts/21_select_heldout.py --verify   # re-derive the selection and compare with the saved manifest
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.data import load_2wiki_parquet, load_trivia_parquet  # noqa: E402
from agentconf.heldout import NEAR_DUP_COSINE, SALT, select_heldout  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, default=[ROOT / "runs/qwen7b", ROOT / "runs/llama8b"])
    ap.add_argument("--splits", default="calibration:200,final:600")
    ap.add_argument("--out", type=Path, default=ROOT / "manifests/heldout_v1/heldout_manifest.json")
    ap.add_argument("--verify", action="store_true", help="recompute and compare with the existing manifest")
    args = ap.parse_args()
    if args.verify and not args.out.exists():
        raise SystemExit(f"{args.out} does not exist: nothing to verify")
    if args.out.exists() and not args.verify:
        raise SystemExit(f"{args.out} exists: the selection is made once")
    import pyarrow.parquet as pq

    specs = json.loads((ROOT / "manifests/submission_v2/p0/dataset_verification.json").read_text(encoding="utf-8"))
    hp, tq = specs["datasets"]["hotpotqa"], specs["datasets"]["triviaqa"]
    wiki = json.loads((ROOT / "manifests/source_swap/data.json").read_text(encoding="utf-8"))["datasets"]["2wikimultihopqa_dev"]
    for spec in (hp, tq):
        if sha256(ROOT / spec["local_path"]) != spec["sha256"]:
            raise SystemExit(f"{spec['local_path']} does not match its recorded sha256")
    wiki_path = ROOT / wiki["local_path"]
    if sha256(wiki_path) != wiki["sha256"]:
        raise SystemExit(f"{wiki_path} does not match its recorded sha256")

    legacy = [json.loads((r / "meta.json").read_text(encoding="utf-8"))["split"] for r in args.runs]
    if any(set(s) != set(legacy[0]) for s in legacy):
        raise SystemExit("legacy question sets differ across runs")
    legacy = set(legacy[0])

    t = pq.read_table(ROOT / hp["local_path"], columns=["id", "question"]).to_pylist()
    pool = [(r["id"], r["question"]) for r in t]
    refs = {"hotpot_legacy": [q for i, q in pool if i in legacy],
            "triviaqa_validation": [r["question"] for r in load_trivia_parquet(ROOT / tq["local_path"]).values()],
            "2wiki_dev": [r["question"] for r in load_2wiki_parquet(wiki_path)]}
    names, sizes = zip(*[(n, int(k)) for n, k in (x.split(":") for x in args.splits.split(","))])
    splits, report = select_heldout(pool, legacy, [q for v in refs.values() for q in v], sizes)

    manifest = {
        "rule": {"module": "agentconf/heldout.py", "salt": SALT, "near_dup_cosine": NEAR_DUP_COSINE,
                 "tfidf": "word 1-2 grams over normalised text, fitted on candidates + references",
                 "order": "sha256(salt + id), first split first"},
        "dataset": {"name": "hotpotqa/hotpot_qa", "config": hp["config"], "split": hp["split"],
                    "revision": hp["revision"], "local_path": hp["local_path"], "sha256": hp["sha256"]},
        "references": {"hotpot_legacy": {"n": len(refs["hotpot_legacy"]), "from": [str(r.relative_to(ROOT)) for r in args.runs]},
                       "triviaqa_validation": {"n": len(refs["triviaqa_validation"]), "revision": tq["revision"]},
                       "2wiki_dev": {"n": len(refs["2wiki_dev"]), "revision": wiki["revision"]}},
        "exclusions": report,
        "splits": dict(zip(names, splits)),
    }
    if args.verify:
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        diff = [k for k in ("rule", "dataset", "exclusions", "splits") if saved.get(k) != manifest[k]]
        print("manifest reproduced" if not diff else f"MISMATCH in {diff}", f"(sha256 {sha256(args.out)})")
        raise SystemExit(1 if diff else 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"exclusions": report, "sizes": {n: len(s) for n, s in zip(names, splits)},
                      "manifest_sha256": sha256(args.out)}, indent=1))


if __name__ == "__main__":
    main()
