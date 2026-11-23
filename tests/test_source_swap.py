"""scripts/20_source_swap.py end to end on a synthetic run directory (random features, no model)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L, D = [0, 1], 6


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _source(dirpath, n, rng, prefix, base):
    y = (rng.random(n) < base).astype(int)
    _jsonl(dirpath / "items.jsonl", [{"id": f"{prefix}{i}", "correct": int(v)} for i, v in enumerate(y)])
    f = rng.standard_normal((n, len(L), D)).astype(np.float16)
    f[:, :, 0] += y[:, None]
    np.savez(dirpath / "feats.npz", mean=f, last=f)
    (dirpath / "meta.json").write_text(json.dumps({"complete": True}))
    return y, f


def test_source_swap_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    run = tmp_path / "run"
    (run / "feats").mkdir(parents=True)
    (run / "results").mkdir()
    eps, steps = [], []
    for q in range(60):
        n = 3
        eps.append({"qid": f"q{q}", "n_steps": n})
        kinds = ["Search", "Search", "Finish"]
        for t in range(1, n + 1):
            c = int(rng.random() < 0.6)
            steps.append({"qid": f"q{q}", "t": t, "kind": kinds[t - 1], "correct": c,
                          "title": f"T{q}-{t}" if kinds[t - 1] == "Search" else None,
                          "split": "test" if q < 40 else "train"})
        f = rng.standard_normal((n, len(L), D)).astype(np.float16)
        np.savez(run / "feats" / f"q{q}.npz", mean=f, last=f)
    _jsonl(run / "episodes.jsonl", eps)
    _jsonl(run / "steps.jsonl", steps)
    (run / "meta.json").write_text(json.dumps({"model": "tiny", "layers": L}))
    y, f = _source(run / "s0tmp", 80, rng, "t", 0.5)
    _jsonl(run / "trivia.jsonl", [{"id": f"t{i}", "correct": int(v)} for i, v in enumerate(y)])
    np.savez(run / "trivia_feats.npz", mean=f, last=f)
    y1, _ = _source(run / "sources" / "trivia_react-main", 80, rng, "t", 0.5)
    y2, _ = _source(run / "sources" / "wiki_search_qa-main", 120, rng, "w", 0.7)
    m = min(min(int(v.sum()), int((1 - v).sum())) for v in (y, y1, y2))

    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "20_source_swap.py"), "--run", str(run),
                          "--pca", "3", "--n-boot", "20", "--min-m", "10"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    tests = json.loads((run / "results" / "source_swap_tests.json").read_text())
    assert tests["m"] == m
    for k in ("I", "H2_distractor_S2QA_minus_S0", "format_C1_minus_S0_finish"):
        assert {"estimate", "ci_lo", "ci_hi"} <= set(tests["tests"][k])
    assert set(tests["readouts"]) == {"S0", "C1", "S2QA"}

    short = subprocess.run([sys.executable, str(ROOT / "scripts" / "20_source_swap.py"), "--run", str(run),
                            "--min-m", "1000"], capture_output=True, text=True)
    assert short.returncode != 0 and "extend S2-QA" in (short.stderr + short.stdout)

    up = subprocess.run([sys.executable, str(ROOT / "scripts" / "20_source_swap.py"), "--run", str(run),
                         "--pca", "3", "--n-boot", "20", "--min-m", "10", "--readout", "upper"],
                        capture_output=True, text=True)
    assert up.returncode == 0, up.stderr
    ut = json.loads((run / "results" / "source_swap_upper_tests.json").read_text())
    assert ut["readout_rule"] == "upper" and ut["readouts"]["S0"] == [["mean", 1], ["last", 1]]
    assert json.loads((run / "results" / "source_swap_tests.json").read_text()) == tests  # default output untouched

    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "20_source_swap.py"), "--run", str(run),
                          "--min-m", "10", "--target-split", "nope"], capture_output=True, text=True)
    assert bad.returncode != 0 and "no steps with split" in (bad.stderr + bad.stdout)
