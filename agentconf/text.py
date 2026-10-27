"""Answer normalisation and matching (SQuAD-style)."""
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def tokens(s):
    return normalize(s).split()


def f1(pred, gold):
    p, g = tokens(pred), tokens(gold)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def matches_alias(pred, golds, max_extra_tokens=5):
    """Exact match with any alias, or an alias appearing as a whole-word span in a short prediction."""
    p = normalize(pred)
    if not p:
        return False
    for g in (normalize(g) for g in golds):
        if not g:
            continue
        if p == g:
            return True
        if re.search(rf"(^| ){re.escape(g)}( |$)", p) and len(p.split()) <= len(g.split()) + max_extra_tokens:
            return True
    return False
