"""Model loading, generation, and teacher-forced feature extraction (hidden states + token log-probs)."""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, load_in_4bit=True, dtype="bfloat16", device_map="auto"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype)
        )
    else:
        kwargs["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


def select_layers(n_hidden, spec="auto"):
    """Indices into output_hidden_states (0 = embeddings, n_hidden-1 = last layer)."""
    if spec == "all":
        return list(range(n_hidden))
    if spec == "auto":
        return sorted({int(round(x)) for x in np.linspace(0, n_hidden - 1, 8)})
    return [int(x) for x in str(spec).split(",")]


@torch.no_grad()
def generate(model, tok, prompt, n=1, do_sample=False, temperature=0.7, max_new_tokens=128, stop_strings=None,
             sample_batch_size=None):
    """Sample in bounded batches so long contexts do not spill GPU memory into system RAM."""
    batch_size = int(sample_batch_size if sample_batch_size is not None
                     else os.environ.get("AGENTCONF_SAMPLE_BATCH_SIZE", "1"))
    if n < 1 or batch_size < 1:
        raise ValueError("n and sample_batch_size must be positive")
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False,
              return_token_type_ids=False).to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
    if stop_strings:
        kw.update(stop_strings=stop_strings, tokenizer=tok)
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    else:
        kw.update(do_sample=False, temperature=None, top_k=None, top_p=None)
    results = []
    total = n if do_sample else 1
    for start in range(0, total, batch_size):
        if do_sample:
            kw["num_return_sequences"] = min(batch_size, total - start)
        if enc["input_ids"].is_cuda:
            torch.cuda.empty_cache()
        out = model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        results.extend(tok.decode(seq, skip_special_tokens=True) for seq in new)
        del new, out
        if enc["input_ids"].is_cuda:
            torch.cuda.empty_cache()
    return results


@torch.no_grad()
def span_features(model, tok, text, span, layers):
    """
    Teacher-forced forward pass over `text`. For the character span [start, end):
      mean   [len(layers), d]  hidden states averaged over the span's tokens
      last   [len(layers), d]  hidden state at the span's last token
      lp_mean, lp_min          log-probs the model assigns to the span's tokens
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True,
              return_token_type_ids=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    start, end = span
    idx = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]
    if not idx:  # span fell inside one token boundary oddity: use the last token
        idx = [len(offsets) - 1]
    enc = {k: v.to(model.device) for k, v in enc.items()}
    # Full-sequence logits would cost ~1 GB at 2k tokens x 152k vocab, so keep one position and
    # project only the positions we need (the last hidden state is already final-normed).
    out = model(**enc, output_hidden_states=True, logits_to_keep=1, use_cache=False)
    hs = out.hidden_states
    sel = torch.stack([hs[l][0, idx, :] for l in layers]).float()  # [L, n_tok, d]
    ids = enc["input_ids"][0]
    pos = [i - 1 for i in idx if i > 0]
    tok_lp = []
    if pos:
        logits = model.get_output_embeddings()(hs[-1][0, pos, :]).float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp[torch.arange(len(pos)), ids[[p + 1 for p in pos]]].tolist()
    return {
        "mean": sel.mean(dim=1).cpu().numpy().astype(np.float16),
        "last": sel[:, -1, :].cpu().numpy().astype(np.float16),
        "lp_mean": float(np.mean(tok_lp)) if tok_lp else float("nan"),
        "lp_min": float(np.min(tok_lp)) if tok_lp else float("nan"),
        "n_tokens": len(idx),
    }
