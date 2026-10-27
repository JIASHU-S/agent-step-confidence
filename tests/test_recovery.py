"""Crash injection at commit boundaries, with no model or dataset downloads."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from agentconf import io
from agentconf.checkpoints import run_lock
from agentconf.runs import load_run

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_torn(path, row):
    with path.open("ab") as f:
        f.write(json.dumps(row).encode()[:12])
    raise KeyboardInterrupt("simulated process interruption")


@pytest.mark.parametrize("tail", [b'{"id":', b'{"id":"\xe4\xb8'])
def test_repair_torn_tail_preserves_backup(tmp_path, tail):
    path = tmp_path / "records.jsonl"
    io.write_jsonl(path, [{"id": "kept"}])
    with path.open("ab") as f:
        f.write(tail)
    with pytest.warns(RuntimeWarning, match="incomplete final line"):
        assert io.recover_jsonl(path) == [{"id": "kept"}]
    assert next(tmp_path.glob("*.bak")).read_bytes() == tail
    io.append_jsonl(path, [{"id": "next"}])
    assert io.read_jsonl(path) == [{"id": "kept"}, {"id": "next"}]


def test_complete_last_record_without_newline_is_kept(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"id": 1}')
    assert io.recover_jsonl(path) == [{"id": 1}]
    io.append_jsonl(path, [{"id": 2}])
    assert io.read_jsonl(path) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("data", [b'{"id":1}\nBAD\n{"id":2}\n', b'{"id":1}\nBAD\n'])
def test_interior_or_terminated_corruption_is_not_silently_removed(tmp_path, data):
    path = tmp_path / "records.jsonl"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="left unchanged"):
        io.recover_jsonl(path)
    assert path.read_bytes() == data
    assert not list(tmp_path.glob("*.bak"))


@pytest.mark.parametrize("kind", ["json", "jsonl", "npz"])
def test_atomic_publish_failure_preserves_previous_file(tmp_path, monkeypatch, kind):
    path = tmp_path / f"data.{kind}"

    def write(value):
        if kind == "npz":
            io.save_npz(path, mean=np.array([value]))
        elif kind == "jsonl":
            io.write_jsonl(path, [{"value": value}])
        else:
            io.write_json(path, {"value": value})

    write(1)
    original = path.read_bytes()

    def crash(*args):
        raise KeyboardInterrupt("before atomic replacement")

    with monkeypatch.context() as m:
        m.setattr(io.os, "replace", crash)
        with pytest.raises(KeyboardInterrupt):
            write(2)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    write(3)


def test_collection_lock_rejects_duplicate_and_releases_after_exception(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with run_lock(tmp_path):
            with pytest.raises(RuntimeError, match="Another collection"):
                with run_lock(tmp_path):
                    pytest.fail("second collector acquired the lock")
            raise KeyboardInterrupt
    with run_lock(tmp_path):
        pass


def test_collection_lock_releases_after_process_is_killed(tmp_path):
    code = ("import sys,time; from pathlib import Path; from agentconf.checkpoints import run_lock; "
            "out=Path(sys.argv[1]); "
            "\nwith run_lock(out):\n (out/'ready').write_text('ready'); time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.02)
        assert (tmp_path / "ready").exists(), "child failed to acquire lock"
        with pytest.raises(RuntimeError, match="Another collection"):
            with run_lock(tmp_path):
                pytest.fail("concurrent process acquired the lock")
        proc.kill()
        # On Windows the venv launcher can exit before its interpreter child has closed its handles.
        proc.communicate(timeout=10)
        with run_lock(tmp_path):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)


@pytest.fixture
def agent(monkeypatch):
    module = script("01_run_agent")
    calls = []
    examples = [{"id": f"q{i}"} for i in range(3)]
    monkeypatch.setattr(module, "load_hotpot", lambda n, seed: examples[:n])
    monkeypatch.setattr(module, "load_model", lambda *a, **kw: (SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=2)), None))

    def episode(model, tok, ex, layers, **kwargs):
        calls.append(ex["id"])
        mean = np.random.normal(size=(2, len(layers), 4)).astype(np.float16)
        last = np.random.normal(size=mean.shape).astype(np.float16)
        ep = {"qid": ex["id"], "n_steps": 2, "finished": True, "success": 1}
        rows = [{"qid": ex["id"], "t": t, "kind": "Search" if t == 1 else "Finish", "correct": 1}
                for t in (1, 2)]
        return ep, rows, mean, last

    monkeypatch.setattr(module, "run_episode", episode)
    return module, calls


def agent_args(out):
    return argparse.Namespace(model="tiny", out=out, n_train=2, n_test=1, seed=17, max_steps=6,
                              k_samples=5, temperature=0.7, layers="auto", no_4bit=True, no_verbal=False)


@pytest.mark.parametrize("boundary", ["feature", "steps_tail", "episode_tail", "after_commit"])
def test_agent_resume_matches_uninterrupted_run(tmp_path, monkeypatch, agent, boundary):
    module, calls = agent
    baseline, interrupted = tmp_path / "baseline", tmp_path / "interrupted"
    module.collect(agent_args(baseline))
    calls.clear()
    original_append, original_save = module.append_jsonl, module.save_npz
    with monkeypatch.context() as m:
        def save(path, **arrays):
            original_save(path, **arrays)
            if boundary == "feature" and path.stem == "q1":
                raise KeyboardInterrupt

        def append(path, rows):
            if rows[0]["qid"] == "q1":
                if boundary == "steps_tail" and path.name == "steps.jsonl":
                    original_append(path, rows[:1])
                    append_torn(path, rows[1])
                if boundary == "episode_tail" and path.name == "episodes.jsonl":
                    append_torn(path, rows[0])
            original_append(path, rows)
            if boundary == "after_commit" and path.name == "episodes.jsonl" and rows[0]["qid"] == "q1":
                raise KeyboardInterrupt

        m.setattr(module, "save_npz", save)
        m.setattr(module, "append_jsonl", append)
        with pytest.raises(KeyboardInterrupt):
            module.collect(agent_args(interrupted))
    calls.clear()
    module.collect(agent_args(interrupted))
    assert calls == (["q2"] if boundary == "after_commit" else ["q1", "q2"])
    assert load_run(baseline)[1:] == load_run(interrupted)[1:]
    assert len(io.read_jsonl(interrupted / "steps.jsonl")) == 6
    for qid in ("q0", "q1", "q2"):
        with np.load(baseline / "feats" / f"{qid}.npz") as a, np.load(interrupted / "feats" / f"{qid}.npz") as b:
            for pooling in ("mean", "last"):
                np.testing.assert_array_equal(a[pooling], b[pooling])


def test_agent_repairs_missing_and_corrupt_features(tmp_path, agent):
    module, calls = agent
    module.collect(agent_args(tmp_path))
    (tmp_path / "feats" / "q1.npz").write_bytes(b"broken zip")
    (tmp_path / "feats" / "q2.npz").unlink()
    calls.clear()
    with pytest.warns(RuntimeWarning, match="missing or damaged"):
        module.collect(agent_args(tmp_path))
    assert calls == ["q1", "q2"]
    assert len(io.read_jsonl(tmp_path / "episodes.jsonl")) == 3
    assert len(io.read_jsonl(tmp_path / "steps.jsonl")) == 6


@pytest.mark.parametrize("field,value", [("model", "other"), ("seed", 20), ("n_train", 1),
                                       ("max_steps", 8), ("no_4bit", False), ("layers", "all")])
def test_agent_parameter_mismatch_does_not_modify_data(tmp_path, agent, field, value):
    module, calls = agent
    args = agent_args(tmp_path)
    module.collect(args)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    calls.clear()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="different settings"):
        module.collect(args)
    assert not calls
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_legacy_agent_run_remains_readable_without_reloading_model(tmp_path, monkeypatch, agent):
    module, calls = agent
    module.collect(agent_args(tmp_path))
    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["layers_spec"], meta["sampling_seed"]
    io.write_json(tmp_path / "meta.json", meta)
    monkeypatch.setattr(module, "load_model", lambda *a, **kw: pytest.fail("finished run loaded a model"))
    calls.clear()
    module.collect(agent_args(tmp_path))
    assert not calls


def test_agent_dataset_change_is_rejected_without_modifying_outputs(tmp_path, monkeypatch, agent):
    module, calls = agent
    args = agent_args(tmp_path)
    module.collect(args)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(module, "load_hotpot", lambda *a, **kw: [{"id": f"other{i}"} for i in range(3)])
    with pytest.raises(ValueError, match="different settings"):
        module.collect(args)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.fixture
def trivia(monkeypatch):
    module = script("02_trivia_features")
    calls = []
    examples = [{"id": f"tv{i}", "question": f"tv{i}", "answers": ["answer"]} for i in range(3)]
    monkeypatch.setattr(module, "load_trivia", lambda n, seed: examples[:n])
    monkeypatch.setattr(module, "load_model", lambda *a, **kw: (None, None))
    monkeypatch.setattr(module, "qa_messages", lambda question: question)
    monkeypatch.setattr(module, "render", lambda tok, messages: messages)
    monkeypatch.setattr(module, "generate", lambda *a, **kw: ["answer"])

    def features(model, tok, text, span, layers):
        calls.append(text.split("answer")[0])
        return {"mean": np.random.normal(size=(len(layers), 4)).astype(np.float16),
                "last": np.random.normal(size=(len(layers), 4)).astype(np.float16), "lp_mean": -1.0}

    monkeypatch.setattr(module, "span_features", features)
    return module, calls


def trivia_args(out):
    io.write_json(out / "meta.json", {"model": "tiny", "no_4bit": True, "layers": [0, 1, 2]})
    return argparse.Namespace(model="tiny", out=out, n=3, seed=17, no_4bit=True)


@pytest.mark.parametrize("boundary", ["feature", "progress_tail", "after_commit", "export"])
def test_trivia_resume_matches_uninterrupted_export(tmp_path, monkeypatch, trivia, boundary):
    module, calls = trivia
    baseline, interrupted = tmp_path / "baseline", tmp_path / "interrupted"
    base_args, args = trivia_args(baseline), trivia_args(interrupted)
    module.collect(base_args)
    original_save, original_append, original_write = module.save_npz, module.append_jsonl, module.write_jsonl
    with monkeypatch.context() as m:
        def save(path, **arrays):
            original_save(path, **arrays)
            if boundary == "feature" and path.stem == "tv1":
                raise KeyboardInterrupt

        def append(path, rows):
            if boundary == "progress_tail" and rows[0]["id"] == "tv1":
                append_torn(path, rows[0])
            original_append(path, rows)
            if boundary == "after_commit" and rows[0]["id"] == "tv1":
                raise KeyboardInterrupt

        def write(path, rows):
            if boundary == "export":
                raise KeyboardInterrupt
            original_write(path, rows)

        m.setattr(module, "save_npz", save)
        m.setattr(module, "append_jsonl", append)
        m.setattr(module, "write_jsonl", write)
        with pytest.raises(KeyboardInterrupt):
            module.collect(args)
    assert json.loads((interrupted / "trivia_meta.json").read_text())["complete"] is False
    calls.clear()
    module.collect(args)
    expected_calls = [] if boundary == "export" else ["tv2"] if boundary == "after_commit" else ["tv1", "tv2"]
    assert calls == expected_calls
    assert json.loads((interrupted / "trivia_meta.json").read_text())["complete"] is True
    assert io.read_jsonl(baseline / "trivia.jsonl") == io.read_jsonl(interrupted / "trivia.jsonl")
    assert [r["id"] for r in io.read_jsonl(interrupted / "trivia.progress.jsonl")] == ["tv0", "tv1", "tv2"]
    with np.load(baseline / "trivia_feats.npz") as a, np.load(interrupted / "trivia_feats.npz") as b:
        for pooling in ("mean", "last"):
            np.testing.assert_array_equal(a[pooling], b[pooling])


def test_trivia_reruns_corrupt_feature_and_rebuilds_missing_export(tmp_path, monkeypatch, trivia):
    module, calls = trivia
    args = trivia_args(tmp_path)
    module.collect(args)
    (tmp_path / "trivia_checkpoints" / "tv1.npz").write_bytes(b"bad")
    calls.clear()
    with pytest.warns(RuntimeWarning, match="missing or damaged"):
        module.collect(args)
    assert calls == ["tv1"]
    (tmp_path / "trivia_feats.npz").unlink()
    (tmp_path / "trivia.jsonl").write_text("broken", encoding="utf-8")
    monkeypatch.setattr(module, "load_model", lambda *a, **kw: pytest.fail("export rebuild loaded a model"))
    calls.clear()
    module.collect(args)
    assert not calls and len(io.read_jsonl(tmp_path / "trivia.jsonl")) == 3
    with np.load(tmp_path / "trivia_feats.npz") as f:
        assert f["mean"].shape == (3, 3, 4)


@pytest.mark.parametrize("field,value", [("model", "other"), ("seed", 2), ("n", 2), ("no_4bit", False)])
def test_trivia_parameter_mismatch_preserves_existing_files(tmp_path, trivia, field, value):
    module, calls = trivia
    args = trivia_args(tmp_path)
    module.collect(args)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    setattr(args, field, value)
    with pytest.raises(ValueError, match="different settings"):
        module.collect(args)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_trivia_dataset_change_is_rejected_without_modifying_outputs(tmp_path, monkeypatch, trivia):
    module, calls = trivia
    args = trivia_args(tmp_path)
    module.collect(args)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(module, "load_trivia", lambda *a, **kw: [{"id": f"other{i}"} for i in range(3)])
    with pytest.raises(ValueError, match="different settings"):
        module.collect(args)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_analysis_rejects_incomplete_trivia_export(tmp_path, monkeypatch):
    module = script("03_step_signals")
    io.write_json(tmp_path / "trivia_meta.json", {"complete": False})
    monkeypatch.setattr(module.sys, "argv", ["03_step_signals.py", "--run", str(tmp_path)])
    monkeypatch.setattr(module, "load_run", lambda *a: pytest.fail("analysis read partial results"))
    with pytest.raises(ValueError, match="incomplete"):
        module.main()
