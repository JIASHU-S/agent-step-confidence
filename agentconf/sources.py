"""Source tasks for the source-swap experiment (protocols/source_swap_v1.md): item selection, prompts and labels.

Model-free so that selection and labelling are unit-tested; scripts/19_source_features.py adds generation and
hidden-state extraction.
"""
import hashlib
import re
import unicodedata

from .env import BM25, HotpotEnv
from .text import normalize, tokens

SOURCES = ("trivia_react", "wiki_search", "wiki_search_qa")
PILOT_RESERVED = 200  # first items of the 2Wiki ordering are reserved for pilots and never used in the main run


def _h(prefix, key):
    return hashlib.sha256(f"{prefix}{key}".encode()).hexdigest()


def hop_of(qid):
    """1 or 2, fixed per question by hash: half the lookup decisions are first-hop, half second-hop."""
    return int(_h("source-swap-v1-hop:", qid)[:8], 16) % 2 + 1


def order_2wiki(examples, exclude_questions=()):
    """Deterministic order of usable 2Wiki questions: at least two supporting titles that are in the context, and no
    normalised-text duplicate of an excluded (target-domain) question."""
    excl = {normalize(q) for q in exclude_questions}
    ok = [ex for ex in examples
          if len(ex["supporting_ordered"]) >= 2 and set(ex["supporting_ordered"]) <= set(ex["titles"])
          and normalize(ex["question"]) not in excl]
    return sorted(ok, key=lambda ex: _h("source-swap-v1-select:", ex["id"]))


def select_2wiki(ordered, source, role, n):
    """wiki_search and wiki_search_qa get disjoint questions (alternating positions); pilots come from the reserved
    head of the ordering, main runs from after it."""
    k = SOURCES.index(source) - 1  # wiki_search -> 0, wiki_search_qa -> 1
    pool = ordered[:PILOT_RESERVED] if role == "pilot" else ordered[PILOT_RESERVED:]
    picked = pool[k::2]
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} {role} questions available for {source}, {n} requested")
    return picked[:n]


_NO_DECOMP = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae",
                            "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i"})


def fold(s):
    """Strip diacritics: 'Ácratas' -> 'Acratas', 'Żuławski' -> 'Zulawski'."""
    s = unicodedata.normalize("NFKD", s.translate(_NO_DECOMP))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


class FoldedEnv(HotpotEnv):
    """HotpotEnv whose Search matching ignores diacritics. 2Wiki titles are often non-English (8.7% of supporting
    titles have non-ASCII characters) and the model often drops accents, which the HotpotQA environment would label
    as 'no matching document' -- an artefact that hardly occurs in the HotpotQA target (1 no-match step per model).
    Labels are otherwise identical to HotpotEnv's."""

    def __init__(self, example, **kw):
        super().__init__(example, **kw)
        self.bm25 = BM25([tokens(fold(t)) * 2 + tokens(fold(p)) for t, p in zip(self.titles, self.paragraphs)])

    def resolve(self, query):
        q = normalize(fold(query))
        if not q:
            return None
        for i, t in enumerate(self.titles):
            if normalize(fold(t)) == q:
                return i
        scores = self.bm25.scores(tokens(fold(query)))
        best = max(range(len(scores)), key=lambda i: scores[i])
        return best if scores[best] > 0 else None


def search_type(env, kind, arg):
    """Label and error type of one action taken in `env` (mutates env like HotpotEnv.step)."""
    if kind == "Search":
        i = env.resolve(arg)
        title = env.titles[i] if i is not None else None
        repeat = title is not None and title in env.retrieved
        obs, lab = env.step(kind, arg)
        if lab["correct"]:
            etype = "search_ok"
        elif title is None:
            etype = "search_nomatch"
        elif repeat:
            etype = "search_repeat"
        else:
            etype = "search_distractor"
        return obs, {"correct": lab["correct"], "title": title, "etype": etype}
    if kind == "Finish":
        obs, lab = env.step(kind, arg)
        return obs, {"correct": lab["correct"], "f1": lab["f1"], "etype": "finish_ok" if lab["correct"] else "finish_wrong"}
    obs, lab = env.step(kind, arg)
    return obs, {"correct": 0, "etype": "invalid"}


def lookup_env(example, hop):
    """Environment and history for one lookup decision. For hop 2 the first supporting paragraph has been read.
    Returns (env, react_history_text, read) where read = [(title, observation)] for the QA-format prompt."""
    env = FoldedEnv(example)
    if hop == 1:
        return env, "", []
    g1 = example["supporting_ordered"][0]
    obs, _ = env.step("Search", g1)
    hist = f"Thought 1: I should look up {g1}.\nAction 1: Search[{g1}]\nObservation 1: {obs}\n"
    text = obs.split("] ", 1)[1] if obs.startswith("[") else obs
    return env, hist, [(g1, text)]


def clean_title_answer(raw):
    """First line of a lookup answer, without surrounding quotes, brackets or a trailing full stop."""
    s = raw.split("\n")[0].strip()
    s = re.sub(r"^(Search\s*)?\[(.*)\]$", r"\2", s, flags=re.IGNORECASE)
    return s.strip(" \"'`*.")


def cut_finish_arg(raw):
    """Generated continuation after 'Finish[': up to the closing bracket or line end."""
    s = raw.split("\n")[0]
    j = s.find("]")
    return (s[:j] if j >= 0 else s).strip()


def balance_indices(ids, y, m):
    """m positives and m negatives, each the first in sha256('source-swap-v1-balance:' + id) order (label-free order;
    labels only decide which list an item joins). Returns sorted indices into ids."""
    order = sorted(range(len(ids)), key=lambda i: _h("source-swap-v1-balance:", ids[i]))
    pos = [i for i in order if y[i] == 1][:m]
    neg = [i for i in order if y[i] == 0][:m]
    if len(pos) < m or len(neg) < m:
        raise ValueError(f"cannot balance to m={m}: {len(pos)} positives, {len(neg)} negatives")
    return sorted(pos + neg)


def source_split(ids, train_share=0.8):
    """Boolean source-train mask: first 80% in sha256('source-swap-v1:' + id) order."""
    order = sorted(range(len(ids)), key=lambda i: _h("source-swap-v1:", ids[i]))
    tr = [False] * len(ids)
    for i in order[: int(round(train_share * len(ids)))]:
        tr[i] = True
    return tr
