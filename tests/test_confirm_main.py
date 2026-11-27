"""scripts/23_confirm_main.py on synthetic runs (no model): freeze, refusals, scoring once, flags and framing."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "23_confirm_main.py"
L, D = [0, 1], 6
FAST = ["--pca", 3, "--n-boot", 30, "--draws", 60, "--tune", 60, "--chains", 2, "--text-encoder", "hash:16"]


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _agent(run, qids, split, rng, meta, pre=True):
    (run / "feats").mkdir(parents=True)
    (run / "feats_pre").mkdir()
    eps, steps = [], []
    for q in qids:
        n = int(rng.integers(2, 5))
        ys = (rng.random(n) < 0.65).astype(int)
        for t in range(1, n + 1):
            kind = "Finish" if t == n else "Search"
            steps.append({"qid": q, "t": t, "kind": kind, "correct": int(ys[t - 1]), "split": split,
                          "arg": f"query {q} {t % 2}", "title": f"T{q}-{t}" if kind == "Search" else None,
                          "obs": f"paragraph {t}", "text": f"Thought {t}: x\nAction {t}: {kind}[a]",
                          "consistency": float(0.6 * ys[t - 1] + 0.4 * rng.random()),
                          "verbalized": float(rng.integers(0, 100)), "lp_mean": float(rng.normal()),
                          "lp_min": float(rng.normal())})
        eps.append({"qid": q, "question": f"which {q} is it", "n_steps": n, "finished": True,
                    "success": int(ys[-1])})
        f = rng.standard_normal((n, len(L), D)).astype(np.float16)
        f[:, :, 0] += 2 * ys[:, None]
        np.savez(run / "feats" / f"{q}.npz", mean=f, last=f)
        if pre:
            np.savez(run / "feats_pre" / f"{q}.npz", pre_step=f, pre_action=f, has_action=np.ones(n, bool))
    _jsonl(run / "episodes.jsonl", eps)
    _jsonl(run / "steps.jsonl", steps)
    (run / "meta.json").write_text(json.dumps({"model": "tiny", "layers": L, "max_steps": 6, **meta}))


def _train_run(src, rng):
    _agent(src, [f"old{i}" for i in range(40)], "train", rng, {})
    ty = (rng.random(80) < 0.5).astype(int)
    _jsonl(src / "trivia.jsonl", [{"id": f"t{i}", "correct": int(v)} for i, v in enumerate(ty)])
    f = rng.standard_normal((80, len(L), D)).astype(np.float16)
    f[:, :, 0] += ty[:, None]
    np.savez(src / "trivia_feats.npz", mean=f, last=f)


def _run(*a):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, a)], capture_output=True, text=True)


def test_freeze_check_and_confirm(tmp_path):
    rng = np.random.default_rng(0)
    final = [f"new{i}" for i in range(40)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"calibration": ["c0"], "final": final}}))
    good = {"revision": "rev1", "ids_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "splits": "final"}
    src, tgt = tmp_path / "src", tmp_path / "tgt"
    _train_run(src, rng)
    _agent(tgt, final, "final", rng, good, pre=False)
    frozen, out = tmp_path / "frozen.json", tmp_path / "res" / "confirm.json"
    common = ["--manifest", manifest, "--frozen", frozen, "--out", out]

    r = _run(*common, "--pairs", f"{src}={tgt}", "--freeze", *FAST)  # pre-action replay missing
    assert r.returncode != 0 and "07_preaction_features" in r.stdout + r.stderr
    for e in json.loads("[" + ",".join((tgt / "episodes.jsonl").read_text().splitlines()) + "]"):
        with np.load(tgt / "feats" / f"{e['qid']}.npz") as z:
            np.savez(tgt / "feats_pre" / f"{e['qid']}.npz", pre_step=z["mean"], pre_action=z["mean"])

    assert _run(*common, "--freeze", "--pairs", f"{src}={tgt}", *FAST).returncode == 0
    assert _run(*common, "--freeze", "--pairs", f"{src}={tgt}", *FAST).returncode != 0  # once only
    assert _run(*common, *FAST).returncode != 0  # settings fixed at freeze

    np.savez(tgt / "feats_pre" / "new0.npz", pre_step=np.zeros((1, 1)), pre_action=np.zeros((1, 1)))
    r = _run(*common)
    assert r.returncode != 0 and "pre-action" in r.stdout + r.stderr and not out.exists()
    with np.load(tgt / "feats" / "new0.npz") as z:
        np.savez(tgt / "feats_pre" / "new0.npz", pre_step=z["mean"], pre_action=z["mean"])

    r = _run(*common)
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.loads(out.read_text())
    s = str(src.resolve())
    assert set(summary["flags"][s]) == {"P1", "P2", "P3", "S1", "S2", "S3", "S5", "S6"}
    p = summary["primary"][s]["P1_probe_minus_consistency"]
    assert abs(p["level"] - (1 - 0.05 / 3)) < 1e-9 and p["ci_lo"] <= p["estimate"] <= p["ci_hi"]
    model = json.loads((tgt / "results" / "main_confirmation_model.json").read_text())
    assert model["n_questions"] == 40 and "rq2" in model["estimates"]
    assert model["text_baselines"]["selected"] in ("handcrafted", "tfidf", "embedding")
    assert model["text_baselines"]["encoder"] == "hash:16@none"
    assert "question_only" in model["readouts"] and "question_only_step_index" in model["signals"]
    assert model["rq3"]["finished"]["n_episodes"] == 40
    assert "Confirmation already completed" in _run(*common).stdout  # scored once

    (tgt / "steps.jsonl").write_text((tgt / "steps.jsonl").read_text() + "\n")
    out.unlink()
    r = _run(*common)  # target changed after freeze
    assert r.returncode != 0 and "changed since the freeze" in r.stdout + r.stderr


def test_framing_rules():
    spec = importlib.util.spec_from_file_location("confirm_main", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yes = dict.fromkeys(["P1", "P2", "P3", "S1", "S2", "S3", "S5", "S6"], True)
    joint, text = mod.framing({"a": yes, "b": {**yes, "P2": False}})
    assert joint["P1"] and not joint["P2"]
    assert any("P2 not confirmed" in x for x in text)
    joint, text = mod.framing({"a": {**yes, "P1": False}, "b": yes})
    assert any("does not replicate" in x for x in text)


def test_official_freeze_rejects_other_encoder(tmp_path):
    r = _run("--freeze", "--text-encoder", "hash:16")
    assert r.returncode != 0 and "registered" in r.stdout + r.stderr


def test_text_contexts_exclude_current_observation():
    sys.path.insert(0, str(ROOT))
    from agentconf.textbase import HashEncoder, contexts, embedding_features

    steps = [{"qid": "a", "t": 1, "text": "Thought 1: x\nAction 1: Search[A]", "obs": "SECRET-ONE"},
             {"qid": "a", "t": 2, "text": "Thought 2: y\nAction 2: Finish[b]", "obs": "Episode finished."}]
    ctx = contexts(steps, {"a": "who is a"})
    assert ctx[0]["history"] == "" and "SECRET-ONE" not in ctx[0]["current"]
    assert "SECRET-ONE" in ctx[1]["history"] and ctx[1]["prev_obs"] == ["SECRET-ONE"]
    feats = embedding_features(ctx, HashEncoder(8))
    assert feats.shape == (2, 3 * 8 + 4) and np.isfinite(feats).all()
