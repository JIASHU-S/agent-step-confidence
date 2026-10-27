"""Scripts 03 and 04 on a synthetic run whose hidden states carry a correctness signal and whose step errors are
coupled. Checks that outputs are written and that the analysis recovers the planted structure."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agentconf.io import write_jsonl

pytest.importorskip("pymc")
ROOT = Path(__file__).resolve().parents[1]
LAYERS = [0, 4, 8]


def make_run(path, n_eps=160, d=24, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    (path / "feats").mkdir(parents=True)
    episodes, steps = [], []
    for e in range(n_eps):
        qid, split, prev_err = f"q{e}", ("train" if e < 110 else "test"), 0
        T = rng.integers(2, 5)
        fm = []
        for t in range(1, T + 1):
            kind = "Finish" if t == T else "Search"
            p = 1 / (1 + np.exp(-(0.8 - 2.5 * prev_err)))  # an earlier error makes the next one likely
            y = int(rng.random() < p)
            h = rng.normal(size=(len(LAYERS), d))
            h[1] += (2 * y - 1) * 0.6 * direction / np.linalg.norm(direction)  # signal only in the middle layer
            fm.append(h.astype(np.float16))
            title = f"T{rng.integers(0, 3)}" if kind == "Search" else None
            steps.append({"qid": qid, "t": t, "kind": kind, "correct": y, "split": split, "title": title,
                          "lp_mean": float(rng.normal(-1 + 0.3 * y)), "lp_min": float(rng.normal(-3)),
                          "consistency": float(rng.random()), "verbalized": float(rng.random())})
            prev_err = 1 - y
        np.savez_compressed(path / "feats" / f"{qid}.npz", mean=np.stack(fm), last=np.stack(fm))
        (path / "feats_pre").mkdir(exist_ok=True)
        pre = np.stack(fm).astype(np.float32) + rng.normal(scale=1.0, size=(T, len(LAYERS), d))
        np.savez_compressed(path / "feats_pre" / f"{qid}.npz", pre_step=rng.normal(size=pre.shape).astype(np.float16),
                            pre_action=pre.astype(np.float16), has_action=np.ones(T, bool))
        episodes.append({"qid": qid, "n_steps": int(T), "finished": True, "success": steps[-1]["correct"],
                         "split": split})
    write_jsonl(path / "episodes.jsonl", episodes)
    write_jsonl(path / "steps.jsonl", steps)
    (path / "meta.json").write_text(json.dumps({"layers": LAYERS}))
    tv = rng.normal(size=(300, len(LAYERS), d))
    ty = rng.integers(0, 2, 300)
    tv[:, 1] += (2 * ty - 1)[:, None] * 1.5 * direction / np.linalg.norm(direction)
    np.savez_compressed(path / "trivia_feats.npz", mean=tv, last=tv)
    write_jsonl(path / "trivia.jsonl", [{"correct": int(v)} for v in ty])


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def test_scripts_03_04(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    subprocess.run([sys.executable, ROOT / "scripts/03_step_signals.py", "--run", run, "--pca", "8",
                    "--few-shot", "30", "--reps", "3"], check=True, capture_output=True)
    res = run / "results"
    assert json.loads((res / "probe_choice.json").read_text())["layer"] == 4
    sig = {(r["signal"], r["kind"]): r for r in rows(res / "rq1_signals.csv")}
    assert float(sig[("probe", "all")]["auroc"]) > 0.65 > float(sig[("consistency", "all")]["auroc"])
    assert len(rows(res / "rq2_transfer.csv")) == 4
    subprocess.run([sys.executable, ROOT / "scripts/04_trajectory.py", "--run", run, "--draws", "300",
                    "--chains", "2"], check=True, capture_output=True)
    loo = json.loads((res / "rq3_loo.json").read_text())
    assert loo["delta_elpd"] > 0 and loo["SR"] > 1  # coupled model preferred on coupled data
    assert {r["method"] for r in rows(res / "rq3_trajectory.csv")} == {"M0_last", "M0_product", "M0_min",
                                                                        "M1_forward"}
    assert len(rows(res / "rq3_handoff.csv")) == 38

    subprocess.run([sys.executable, ROOT / "scripts/06_robustness.py", "--run", run, "--n-boot", "200",
                    "--pca", "8"], check=True, capture_output=True)
    rb = {r["comparison"]: r for r in rows(res / "robust_rq1_bootstrap.csv")}
    assert float(rb["probe - consistency"]["ci_low"]) > 0
    assert {r["signal"] for r in rows(res / "robust_lenient_rq1.csv")} >= {"probe", "consistency"}
    r3 = json.loads((res / "robust_rq3.json").read_text())
    assert r3["aurc_M0_minus_M1"]["ci_low"] <= r3["aurc_M0_minus_M1"]["mean"] <= r3["aurc_M0_minus_M1"]["ci_high"]
    assert (res / "figures" / "handoff_bands.png").exists()

    subprocess.run([sys.executable, ROOT / "scripts/08_preaction_analysis.py", "--run", run, "--n-boot", "200",
                    "--pca", "8"], check=True, capture_output=True)
    pa = {r["point"]: r for r in rows(res / "preaction.csv")}
    assert float(pa["pre_action"]["auroc"]) > float(pa["pre_step"]["auroc"])  # planted: only pre_action has signal
    assert abs(float(pa["pre_step"]["auroc"]) - 0.5) < 0.1
