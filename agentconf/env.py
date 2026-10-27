"""Offline retrieval environment over the 10 paragraphs HotpotQA (distractor setting) gives each question.

Actions
    Search[query]   returns the best-matching paragraph (exact title match first, then BM25 over title + text)
    Finish[answer]  ends the episode

Step labels (automatic, no judge model)
    Search  correct if it returns a supporting paragraph the agent has not retrieved before
    Finish  correct if token F1 with the gold answer >= finish_f1
    invalid action (no parsable Search[...] / Finish[...]) is always incorrect
"""
import math
import re
from collections import Counter

from .text import f1, normalize, tokens

ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*(Search|Finish)\s*\[([^\]\n]*)\]", re.IGNORECASE)


def parse_action(step_text):
    """Returns (kind, argument, end_char) for the first action in the text, or ("invalid", "", None)."""
    m = ACTION_RE.search(step_text)
    if not m:
        return "invalid", "", None
    return m.group(1).capitalize(), m.group(2).strip(), m.end()


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / max(len(docs), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class HotpotEnv:
    def __init__(self, example, max_obs_words=120, finish_f1=0.5):
        self.ex = example
        self.titles = list(example["titles"])
        self.paragraphs = [" ".join(s.strip() for s in sents) for sents in example["sentences"]]
        self.gold_titles = set(example["supporting_titles"])
        self.max_obs_words = max_obs_words
        self.finish_f1 = finish_f1
        # Titles count twice so that a query naming an entity prefers that entity's own paragraph.
        self.bm25 = BM25([tokens(t) * 2 + tokens(p) for t, p in zip(self.titles, self.paragraphs)])
        self.reset()

    def reset(self):
        self.retrieved = set()
        self.done = False
        self.answer = None

    def resolve(self, query):
        """Index of the paragraph Search[query] would return (no state change), or None."""
        q = normalize(query)
        if not q:
            return None
        for i, t in enumerate(self.titles):
            if normalize(t) == q:
                return i
        scores = self.bm25.scores(tokens(query))
        best = max(range(len(scores)), key=lambda i: scores[i])
        return best if scores[best] > 0 else None

    def canonical(self, kind, arg):
        """Equivalence key used for sampling consistency: two searches agree if they return the same paragraph."""
        if kind == "Search":
            i = self.resolve(arg)
            return f"S:{self.titles[i]}" if i is not None else "S:<none>"
        if kind == "Finish":
            return f"F:{normalize(arg)}"
        return "INVALID"

    def step(self, kind, arg):
        """Returns (observation, label dict)."""
        if kind == "Search":
            i = self.resolve(arg)
            if i is None:
                return "No matching document.", {"correct": 0, "title": None}
            title = self.titles[i]
            new_gold = title in self.gold_titles and title not in self.retrieved
            self.retrieved.add(title)
            words = self.paragraphs[i].split()
            text = " ".join(words[: self.max_obs_words]) + (" ..." if len(words) > self.max_obs_words else "")
            return f"[{title}] {text}", {"correct": int(new_gold), "title": title}
        if kind == "Finish":
            self.done = True
            self.answer = arg
            score = f1(arg, self.ex["answer"])
            return "Episode finished.", {"correct": int(score >= self.finish_f1), "f1": score}
        return "Invalid action. Use Search[...] or Finish[...].", {"correct": 0}
