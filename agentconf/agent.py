"""ReAct episode runner that records, for every step, the label and four confidence signals:

    probe features   hidden states over the Action span (mean-pooled and last token)
    logprob          mean / min token log-prob of the Action span
    consistency      share of k sampled alternative steps whose action is equivalent to the greedy one
    verbalized       the model's own 0-100 rating of the step, from a separate prompt
"""
import re

import numpy as np

from .env import HotpotEnv, parse_action
from .model import generate, span_features
from .prompts import react_messages, render, verbal_messages

STOP = ["\nObservation", "Observation "]


def _cut_step(text):
    """Keep one Thought + Action: cut at the end of the first action, else at the first Observation."""
    kind, arg, end = parse_action(text)
    if end is not None:
        return text[:end], kind, arg
    for s in STOP:
        j = text.find(s)
        if j >= 0:
            text = text[:j]
    return text.rstrip(), "invalid", ""


def _parse_score(text):
    # Skip step references such as "Action 2" or "step 3"; some models restate the question before the score.
    for m in re.finditer(r"\d+(\.\d+)?", text):
        if re.search(r"(action|step|thought|observation)\s*$", text[:m.start()], re.IGNORECASE):
            continue
        return float(min(max(float(m.group()), 0.0), 100.0)) / 100.0
    return float("nan")


def run_episode(model, tok, example, layers, max_steps=6, k_samples=5, temperature=0.7, max_new_tokens=128,
                verbal=True, progress=None):
    env = HotpotEnv(example)
    prompt = render(tok, react_messages(example))
    traj = ""  # assistant-side transcript so far
    steps, feats_mean, feats_last = [], [], []
    for t in range(1, max_steps + 1):
        prefix = prompt + traj + f"Thought {t}:"
        if progress:
            progress(t, "generating action")
        raw = generate(model, tok, prefix, max_new_tokens=max_new_tokens, stop_strings=STOP)[0]
        step_text, kind, arg = _cut_step(f"Thought {t}:" + raw)

        # Action span inside the step (the whole step if no action parsed).
        m = re.search(r"Action\s*\d*\s*:", step_text)
        a0 = m.start() if (m and kind != "invalid") else 0
        full = prompt + traj + step_text
        base = len(prompt + traj)
        if progress:
            progress(t, "extracting features")
        f = span_features(model, tok, full, (base + a0, base + len(step_text)), layers)

        consistency = float("nan")
        if k_samples > 0:
            if progress:
                progress(t, f"sampling {k_samples} alternatives")
            alts = generate(model, tok, prefix, n=k_samples, do_sample=True, temperature=temperature,
                            max_new_tokens=max_new_tokens, stop_strings=STOP)
            key = env.canonical(kind, arg)
            keys = [env.canonical(*_cut_step(f"Thought {t}:" + a)[1:]) for a in alts]
            consistency = float(np.mean([k == key for k in keys]))

        obs, label = env.step(kind, arg)
        traj_with_step = traj + step_text + "\n"
        verbalized, verbal_raw = float("nan"), ""
        if verbal:
            if progress:
                progress(t, "rating confidence")
            vprompt = render(tok, verbal_messages(example, traj_with_step, t))
            verbal_raw = generate(model, tok, vprompt, max_new_tokens=24)[0]
            verbalized = _parse_score(verbal_raw)

        steps.append({
            "qid": example["id"], "t": t, "kind": kind, "arg": arg, "text": step_text, "obs": obs,
            "correct": label["correct"], "title": label.get("title"), "f1": label.get("f1"),
            "lp_mean": f["lp_mean"], "lp_min": f["lp_min"], "n_tokens": f["n_tokens"],
            "consistency": consistency, "verbalized": verbalized, "verbal_raw": verbal_raw,
        })
        feats_mean.append(f["mean"])
        feats_last.append(f["last"])
        traj = traj_with_step + f"Observation {t}: {obs}\n"
        if env.done:
            break

    episode = {
        "qid": example["id"], "question": example["question"], "gold": example["answer"],
        "type": example["type"], "level": example["level"], "n_steps": len(steps),
        "finished": env.done, "answer": env.answer,
        "success": int(bool(env.done and steps[-1]["correct"])),
        "transcript": traj,
    }
    return episode, steps, np.stack(feats_mean), np.stack(feats_last)
