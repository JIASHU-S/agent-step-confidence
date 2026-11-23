"""scripts/22_confirm_source_swap.py: freeze, refusal paths, and the framing map, on synthetic runs (no model)."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "22_confirm_source_swap.py"
L, D = [0, 1], 6


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _labelled(dirpath, ids, rng, base, items="items.jsonl", feats="feats.npz"):
    y = (rng.random(len(ids)) < base).astype(int)
    _jsonl(dirpath / items, [{"id": i, "correct": int(v)} for i, v in zip(ids, y)])
    f = rng.standard_normal((len(ids), len(L), D)).astype(np.float16)
    f[:, :, 0] += y[:, None]
    np.savez(dirpath / feats, mean=f, last=f)


def _agent(run, qids, split, rng, extra_meta):
    (run / "feats").mkdir(parents=True)
    eps, steps = [], []
    for q in qids:
        kinds = ["Search", "Search", "Finish"]
        eps.append({"qid": q, "n_steps": 3})
        for t in (1, 2, 3):
            steps.append({"qid": q, "t": t, "kind": kinds[t - 1], "correct": int(rng.random() < 0.6),
                          "title": f"T{q}-{t}" if t < 3 else None, "split": split})
        f = rng.standard_normal((3, len(L), D)).astype(np.float16)
        np.savez(run / "feats" / f"{q}.npz", mean=f, last=f)
    _jsonl(run / "episodes.jsonl", eps)
    _jsonl(run / "steps.jsonl", steps)
    (run / "meta.json").write_text(json.dumps({"model": "tiny", "layers": L, **extra_meta}))


def _source_run(src, rng):
    _agent(src, [f"old{i}" for i in range(5)], "test", rng, {})
    _labelled(src, [f"t{i}" for i in range(80)], rng, 0.5, "trivia.jsonl", "trivia_feats.npz")
    for d, prefix, base in (("trivia_react-main", "t", 0.5), ("wiki_search_qa-main", "w", 0.7)):
        _labelled(src / "sources" / d, [f"{prefix}{i}" for i in range(80 if prefix == "t" else 120)], rng, base)
        (src / "sources" / d / "meta.json").write_text(json.dumps({"complete": True, "resolved_revision": "rev1"}))


def _run(*a):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, a)], capture_output=True, text=True)


FAST = ["--pca", 3, "--n-boot", 20, "--min-m", 10]


def test_freeze_check_and_confirm(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    final = [f"new{i}" for i in range(40)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"calibration": ["c0"], "final": final}}))
    srcs, tgts = [tmp_path / "srcA", tmp_path / "srcB"], [tmp_path / "tgtA", tmp_path / "tgtB"]
    for src in srcs:
        _source_run(src, rng)
    frozen, out = tmp_path / "frozen.json", tmp_path / "res" / "confirm.json"
    common = ["--manifest", manifest, "--frozen", frozen, "--out", out]
    pairs = ["--pairs", *[f"{s}={t}" for s, t in zip(srcs, tgts)]]
    assert _run(*common, *pairs, "--freeze", *FAST).returncode == 0
    assert _run(*common, *pairs, "--freeze").returncode != 0  # once only
    r = _run(*common, "--pairs", f"{srcs[0]}={tgts[0]}", *FAST)  # no single-model confirmation
    assert r.returncode != 0 and "frozen model set" in r.stderr + r.stdout

    good = {"revision": "rev1", "ids_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "splits": "final"}
    _agent(tgts[0], final, "final", rng, good)
    r = _run(*common, *FAST)
    assert r.returncode != 0 and "fixed at freeze" in r.stderr
    r = _run(*common)  # second model not run yet
    assert r.returncode != 0 and "without an episode" in r.stderr + r.stdout and not out.exists()

    _agent(tgts[1], final[:-1] + ["stray"], "final", rng, good)  # one missing, one extra
    r = _run(*common)
    assert r.returncode != 0 and "1 manifest questions without an episode, 1 extra" in r.stderr + r.stdout
    shutil.rmtree(tgts[1])

    _agent(tgts[1], final, "final", rng, {**good, "revision": "other"})
    r = _run(*common)
    assert r.returncode != 0 and "revision" in r.stderr + r.stdout
    (tgts[1] / "meta.json").write_text(json.dumps({"model": "tiny", "layers": L, **good}))

    (tgts[1] / "feats" / f"{final[5]}.npz").write_bytes(b"corrupt")  # damaged features in the second model
    r = _run(*common)
    assert r.returncode != 0 and "unreadable features" in r.stderr + r.stdout
    assert not (tgts[0] / "results").exists() and not out.exists()  # nothing scored for the first model either
    f = rng.standard_normal((3, len(L), D)).astype(np.float16)
    np.savez(tgts[1] / "feats" / f"{final[5]}.npz", mean=f, last=f)

    trivia = srcs[0] / "trivia.jsonl"
    original = trivia.read_bytes()
    trivia.write_bytes(original + b" ")
    r = _run(*common)
    assert r.returncode != 0 and "source file changed" in r.stderr
    trivia.write_bytes(original)

    spec = importlib.util.spec_from_file_location("confirm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old_run = subprocess.run
    def interrupt_second(cmd, **kw):
        if str(tgts[1]) in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "synthetic interruption")
        return old_run(cmd, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", [str(SCRIPT), *map(str, common)])
        patch.setattr(subprocess, "run", interrupt_second)
        with pytest.raises(SystemExit, match="synthetic interruption"):
            mod.main()
    staging, = out.parent.glob(".source_swap_confirmation_*")
    assert (staging / "model0/.selected.complete").exists() and not out.exists()
    assert not (tgts[0] / "results").exists()
    with monkeypatch.context() as patch:
        def resume_second(cmd, **kw):
            assert str(tgts[0]) not in cmd, "completed model was scored again"
            return old_run(cmd, **kw)
        patch.setattr(sys, "argv", [str(SCRIPT), *map(str, common)])
        patch.setattr(subprocess, "run", resume_second)
        mod.main()
    stamp = out.stat().st_mtime_ns
    r = _run(*common)
    assert r.returncode == 0, r.stderr
    assert "already completed" in r.stdout and out.stat().st_mtime_ns == stamp
    res = json.loads(out.read_text())
    assert len(res["models"]) == 2 and set(res["I_G_C"][res["models"][0]]) == {"I", "G", "C"}
    assert {"readouts", "I", "G", "C"} <= set(res["v1_1_selected_rule"][res["models"][1]])
    for t in tgts:
        assert (t / "results" / "source_swap_upper_tests.json").exists() and (t / "results" / "source_swap_tests.json").exists()
    assert not staging.exists()


def test_framing_map():
    spec = importlib.util.spec_from_file_location("confirm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def f(h1, gain, noninf, cost):
        return {"H1_this_model": h1, "gain_this_model": gain, "finish_noninferior_this_model": noninf,
                "finish_cost_this_model": cost}
    assert mod.framing({"a": f(1, 1, 1, 0), "b": f(1, 1, 1, 0)})[0] == "generalization"
    assert mod.framing({"a": f(1, 1, 0, 1), "b": f(1, 1, 0, 1)})[0] == "tradeoff"
    assert mod.framing({"a": f(1, 1, 1, 0), "b": f(1, 1, 0, 1)})[0] == "gain_only"
    assert mod.framing({"a": f(1, 1, 0, 1), "b": f(1, 0, 0, 1)})[0] == "interaction_only"
    assert mod.framing({"a": f(0, 1, 0, 0), "b": f(1, 1, 0, 0)})[0] == "gain_without_interaction"
    assert mod.framing({"a": f(1, 0, 0, 0), "b": f(0, 1, 0, 0)})[0] == "neither_confirmed"
