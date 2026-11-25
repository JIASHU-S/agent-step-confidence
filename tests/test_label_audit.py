"""scripts/24_label_audit.py on a synthetic run: strata, blinding, sample-once, and scoring."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "24_label_audit.py"


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _run_dir(run, model, rng, n=120):
    run.mkdir(parents=True)
    eps, steps = [], []
    for i in range(n):
        q = f"{model}-q{i}"
        n_steps = int(rng.integers(1, 5))
        titles = []
        for t in range(1, n_steps + 1):
            if t == n_steps:
                f1 = float(rng.choice([0.0, 0.3, 0.7, 1.0]))
                s = {"kind": "Finish", "arg": "ans", "f1": f1, "correct": int(f1 >= 0.5), "title": None, "obs": ""}
            elif rng.random() < 0.05:
                s = {"kind": "invalid", "arg": "", "f1": None, "correct": 0, "title": None, "obs": "Invalid."}
            else:
                title = rng.choice(titles) if titles and rng.random() < 0.3 else f"P{int(rng.integers(0, 10))}"
                s = {"kind": "Search", "arg": f"query {t}", "title": title, "f1": None, "obs": f"text of {title}",
                     "correct": int(title not in titles and rng.random() < 0.6)}
                titles.append(title)
            steps.append({"qid": q, "t": t, "split": "train", "text": f"Thought {t}: x\nAction {t}: y", **s})
        eps.append({"qid": q, "question": f"Q{i}?", "gold": "ans", "n_steps": n_steps, "finished": True,
                    "success": steps[-1]["correct"] if steps[-1]["kind"] == "Finish" else 0})
    _jsonl(run / "episodes.jsonl", eps)
    _jsonl(run / "steps.jsonl", steps)
    (run / "meta.json").write_text(json.dumps({"model": model, "layers": [0]}))


def _run(*a):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, a)], capture_output=True, text=True)


def _fill(out, dest, key, flip=()):
    dest.mkdir()
    kind = {r["item_id"]: r for r in key["steps"]}
    code = {("Search", 1): "S-USEFUL", ("Search", 0): "S-UNHELPFUL", ("Finish", 1): "F-CORRECT",
            ("Finish", 0): "F-WRONG", ("invalid", 0): "I-INVALID"}
    for name, idcol in (("annotate_steps.csv", "item_id"), ("annotate_episodes.csv", "episode_id")):
        with (out / name).open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if idcol == "item_id":
                k = kind[r["item_id"]]
                label = 1 - k["auto_label"] if r["item_id"] in flip and k["kind"] != "invalid" else k["auto_label"]
                r["code"] = code[(k["kind"], label)]
            else:
                r["code"] = "E-CORRECT"
        with (dest / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def test_sample_blinding_and_score(tmp_path):
    rng = np.random.default_rng(0)
    runs = [tmp_path / "a", tmp_path / "b"]
    _run_dir(runs[0], "model-A", rng)
    _run_dir(runs[1], "model-B", rng)
    out = tmp_path / "audit"
    r = _run("sample", "--runs", *runs, "--out", out)
    assert r.returncode == 0, r.stderr
    assert _run("sample", "--runs", *runs, "--out", out).returncode != 0  # drawn once

    key = json.loads((out / "key.json").read_text())
    header = (out / "annotate_steps.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    for leak in ("auto_label", "correct", "f1", "model", "stratum", "supporting"):
        assert leak not in header
    sheet = (out / "annotate_steps.csv").read_text(encoding="utf-8-sig")
    assert "model-A" not in sheet and "model-B" not in sheet
    strat = [x for x in key["steps"] if x["source"] == "stratified"]
    assert len({(x["model"], x["qid"], x["t"]) for x in key["steps"]}) == len(key["steps"])
    ep_q = {(e["model"], e["qid"]) for e in key["episodes"]}
    assert len(key["episodes"]) == 40 and not any((x["model"], x["qid"]) in ep_q for x in strat)
    for x in strat:  # strata follow the labels
        assert x["stratum"].startswith({"Search": "search", "Finish": "finish", "invalid": "invalid"}[x["kind"]])
        assert x["auto_label"] == int(x["stratum"] in ("search_correct", "finish_correct_exact",
                                                        "finish_correct_partial"))

    _fill(out, out / "A", key)
    flip = {x["item_id"] for x in strat[:10]}
    _fill(out, out / "B", key, flip=flip)
    r = _run("score", "--out", out, "--annotator", f"A={out / 'A'}", "--annotator", f"B={out / 'B'}")
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "audit_results.json").read_text())
    assert all(abs(v - 1) < 1e-9 for v in res["annotators"]["A"]["strict"]["weighted_accuracy"].values())
    assert res["inter_annotator"]["n"] == len(key["steps"])
    assert set(res["inter_annotator"]["disagreements"]) == {i for i in flip
                                                            if next(x for x in strat if x["item_id"] == i)["kind"]
                                                            != "invalid"}
    assert 0 < res["inter_annotator"]["kappa_codes"] < 1
