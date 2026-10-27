"""Tests for the follow-up scripts 06 (robustness) and 07 (pre-action replay)."""
import numpy as np
import pytest

from tests.test_core import EX
from tests.test_model_tiny import _models, tok  # noqa: F401  (tok is a fixture)
from agentconf.agent import run_episode


def test_lenient_labels():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("rob", Path(__file__).resolve().parents[1] / "scripts/06_robustness.py")
    rob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rob)
    steps = [{"qid": "a", "kind": "Search", "correct": 1, "title": "X"},
             {"qid": "a", "kind": "Search", "correct": 0, "title": "X"},   # repeat of a gold paragraph
             {"qid": "a", "kind": "Search", "correct": 0, "title": "Y"},   # distractor
             {"qid": "b", "kind": "Search", "correct": 0, "title": "X"},   # X is gold only in episode a
             {"qid": "a", "kind": "Finish", "correct": 0, "title": None}]
    assert rob.lenient_labels(steps).tolist() == [1, 1, 0, 0, 0]


def test_preaction_replay_matches_recorded_features(tok, monkeypatch, tmp_path):
    """Record a scripted episode, replay it with script 07, and check the transcript reconstruction is exact:
    the Action-span mean recomputed during replay equals the stored feature."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import agentconf.agent as agent
    from agentconf.io import write_jsonl

    script = {1: " I need Kyoto.\nAction 1: Search[Kyoto University]",
              2: " Now Tohoku.\nAction 2: Search[Tohoku University]",
              3: " 1897 < 1907.\nAction 3: Finish[Kyoto University]"}

    def fake_generate(model, tok_, prompt, n=1, do_sample=False, **kw):
        if "How confident" in prompt:
            return ["90"]
        t = int(prompt.rsplit("Thought ", 1)[1].split(":")[0])
        return [script[t]] * (n if do_sample else 1)

    monkeypatch.setattr(agent, "generate", fake_generate)
    model = list(_models(tok))[1]  # Qwen2 architecture
    layers = [0, 1, 2]
    ep, steps, fm, fl = run_episode(model, tok, EX, layers=layers, k_samples=2)
    run = tmp_path / "run"
    (run / "feats").mkdir(parents=True)
    np.savez_compressed(run / "feats" / "q1.npz", mean=fm, last=fl)
    ep["split"] = "train"
    for s in steps:
        s["split"] = "train"
    write_jsonl(run / "episodes.jsonl", [ep])
    write_jsonl(run / "steps.jsonl", steps)
    (run / "meta.json").write_text(json.dumps({"layers": layers, "model": "tiny", "n_train": 1, "n_test": 0,
                                               "seed": 0}))

    spec = importlib.util.spec_from_file_location(
        "pre", Path(__file__).resolve().parents[1] / "scripts/07_preaction_features.py")
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)
    monkeypatch.setattr(pre, "load_hotpot", lambda n, seed: [EX])
    monkeypatch.setattr(pre, "load_model", lambda name, load_in_4bit: (model, tok))
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    monkeypatch.setattr(sys, "argv", ["07", "--run", str(run), "--verify", "1"])
    pre.main()
    out = np.load(run / "feats_pre" / "q1.npz")
    assert out["pre_step"].shape == out["pre_action"].shape == (3, 3, 32) and out["has_action"].all()
    cos = [float(line.split("min=")[1]) for line in printed if line.startswith("verify")]
    assert len(cos) == 3 and min(cos) > 0.999
    assert not np.allclose(out["pre_step"], out["pre_action"])
    out.close()

    monkeypatch.setattr(pre, "load_model", lambda *a, **k: pytest.fail("complete replay loaded a model"))
    pre.main()


def test_preaction_resume_after_interruption_and_corrupt_checkpoint(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "pre_resume", Path(__file__).resolve().parents[1] / "scripts/07_preaction_features.py")
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)
    meta = {"layers": [0, 1], "model": "tiny", "n_train": 2, "n_test": 0, "seed": 0, "no_4bit": False}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    episodes = [{"qid": qid, "n_steps": 1} for qid in ("q0", "q1")]
    steps = [{"qid": e["qid"], "t": 1, "text": "Thought 1: done\nAction 1: Finish[answer]",
              "kind": "Finish", "obs": "finished"} for e in episodes]
    monkeypatch.setattr(pre, "load_run", lambda run: (meta, episodes, steps))
    monkeypatch.setattr(pre, "load_hotpot", lambda *a, **k: [{"id": e["qid"]} for e in episodes])
    monkeypatch.setattr(pre, "react_messages", lambda example: example["id"])
    monkeypatch.setattr(pre, "render", lambda tok, messages: messages)
    monkeypatch.setattr(pre, "load_model", lambda *a, **k: (None, None))
    replayed = []

    def states(model, tok, full, chars, layers):
        replayed.append(full[:2])
        return np.ones((2, 2, 4), dtype=np.float32), [], None

    monkeypatch.setattr(pre, "step_states", states)
    save = pre.save_npz

    def interrupted_save(path, **arrays):
        if path.stem == "q1":
            raise KeyboardInterrupt("before question checkpoint")
        save(path, **arrays)

    args = SimpleNamespace(run=tmp_path, no_4bit=False, verify=0)
    with monkeypatch.context() as patch:
        patch.setattr(pre, "save_npz", interrupted_save)
        with pytest.raises(KeyboardInterrupt):
            pre.collect(args)
    completed = tmp_path / "feats_pre/q0.npz"
    original = completed.read_bytes()
    replayed.clear()
    pre.collect(args)
    assert replayed == ["q1"] and completed.read_bytes() == original

    completed.write_bytes(b"truncated checkpoint")
    replayed.clear()
    pre.collect(args)
    assert replayed == ["q0"] and pre.valid_checkpoint(completed, 1, 2)

    monkeypatch.setattr(pre, "load_model", lambda *a, **k: pytest.fail("complete replay loaded a model"))
    replayed.clear()
    pre.collect(args)
    assert not replayed
