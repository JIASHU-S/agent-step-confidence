"""Tiny random models with a tokenizer trained on the spot (no downloads)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen2Config,  # noqa: E402
                          Qwen2ForCausalLM)

from agentconf.agent import run_episode  # noqa: E402
from agentconf.model import generate, span_features  # noqa: E402
from tests.test_core import EX  # noqa: E402

CORPUS = ["Thought 1: I need to search.\nAction 1: Search[Kyoto University]\nObservation 1: [Kyoto University] "
          "founded 1897.\nThought 2: done.\nAction 2: Finish[Kyoto University]\n", "Documents: Question: 0 50 100\n"]


@pytest.fixture(scope="module")
def tok():
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(CORPUS * 50, trainers.BpeTrainer(vocab_size=400, special_tokens=["<eos>"],
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>", pad_token="<eos>")
    tk.padding_side = "left"
    return tk


def _models(tok):
    torch.manual_seed(0)
    yield GPT2LMHeadModel(GPT2Config(vocab_size=tok.vocab_size, n_positions=4096, n_embd=32, n_layer=2, n_head=2,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)).eval()
    yield Qwen2ForCausalLM(Qwen2Config(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                       max_position_embeddings=4096, eos_token_id=tok.eos_token_id,
                                       pad_token_id=tok.pad_token_id)).eval()


@torch.no_grad()
def test_span_logprobs_match_full_logits(tok):
    text = "Thought 1: I need to search.\nAction 1: Search[Kyoto University]"
    start = text.index("Action")
    for model in _models(tok):
        f = span_features(model, tok, text, (start, len(text)), layers=[0, 1, 2])
        assert f["mean"].shape == (3, 32) and f["last"].shape == (3, 32)
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True,
                  return_token_type_ids=False)
        offs = enc.pop("offset_mapping")[0].tolist()
        idx = [i for i, (a, b) in enumerate(offs) if b > start and b > a]
        lp = torch.log_softmax(model(**enc).logits[0].float(), -1)
        ids = enc["input_ids"][0]
        ref = [lp[i - 1, ids[i]].item() for i in idx]
        assert np.isclose(f["lp_mean"], np.mean(ref), atol=1e-4), type(model).__name__
        assert np.isclose(f["lp_min"], np.min(ref), atol=1e-4)


def test_run_episode_end_to_end(tok):
    for model in _models(tok):
        ep, steps, fm, fl = run_episode(model, tok, EX, layers=[0, 2], max_steps=3, k_samples=2, max_new_tokens=20)
        assert 1 <= ep["n_steps"] == len(steps) <= 3 and fm.shape == (len(steps), 2, 32) == fl.shape
        for s in steps:
            assert s["correct"] in (0, 1) and 0 <= s["consistency"] <= 1 and np.isfinite(s["lp_mean"])
            assert s["kind"] in ("Search", "Finish", "invalid")
        assert ep["success"] in (0, 1)


def test_sampling_preserves_count_and_seed_with_bounded_batches(tok, monkeypatch):
    model = list(_models(tok))[-1]
    batches = []
    original = model.generate

    def tracked_generate(*args, **kwargs):
        batches.append(kwargs.get("num_return_sequences", 1))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "generate", tracked_generate)
    torch.manual_seed(17)
    first = generate(model, tok, "Thought 1:", n=5, do_sample=True, max_new_tokens=4, sample_batch_size=2)
    assert len(first) == 5 and batches == [2, 2, 1]
    batches.clear()
    torch.manual_seed(17)
    assert generate(model, tok, "Thought 1:", n=5, do_sample=True, max_new_tokens=4, sample_batch_size=2) == first
    assert batches == [2, 2, 1]
    batches.clear()
    generate(model, tok, "Thought 1:", n=5, do_sample=True, max_new_tokens=4)
    assert batches == [1] * 5


def test_scripted_episode_labels(tok, monkeypatch):
    """Scripted generations exercise the env flow: two gold searches, then a correct Finish."""
    import agentconf.agent as agent

    script = {1: " I need Kyoto.\nAction 1: Search[Kyoto University]\nObservation 1: x",
              2: " Now Tohoku.\nAction 2: Search[Tohoku University]",
              3: " 1897 < 1907.\nAction 3: Finish[Kyoto University]"}

    def fake_generate(model, tok_, prompt, n=1, do_sample=False, **kw):
        if "How confident" in prompt:
            return ["90"]
        t = int(prompt.rsplit("Thought ", 1)[1].split(":")[0])
        return [script[t]] * n if do_sample else [script[t]]

    monkeypatch.setattr(agent, "generate", fake_generate)
    model = next(_models(tok))
    progress = []
    ep, steps, fm, _ = run_episode(model, tok, EX, layers=[0, 2], max_steps=6, k_samples=3,
                                  progress=lambda t, phase: progress.append((t, phase)))
    assert [s["kind"] for s in steps] == ["Search", "Search", "Finish"]
    assert [s["correct"] for s in steps] == [1, 1, 1] and ep["success"] == 1 and ep["finished"]
    assert all(s["consistency"] == 1.0 and s["verbalized"] == 0.9 for s in steps)
    assert "Observation 1: [Kyoto University]" in ep["transcript"] and "Observation 1: x" not in ep["transcript"]
    assert [t for t, phase in progress if phase == "generating action"] == [1, 2, 3]
    assert [t for t, phase in progress if phase == "rating confidence"] == [1, 2, 3]
