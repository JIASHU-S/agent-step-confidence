"""Step error types and position-controlled AUROC for the transfer decomposition (scripts/18)."""
import numpy as np
from scipy.stats import rankdata

TYPES = ("search_ok", "search_repeat", "search_distractor", "search_nomatch", "finish_ok", "finish_wrong", "invalid")


def error_types(steps):
    """One type per step, following the environment's label rule (agentconf/env.py).

    A Search is correct when it returns a supporting paragraph not retrieved before in the episode. A wrong Search is
    a repeat (its paragraph was already retrieved in this episode), a distractor (a new paragraph outside the
    supporting set) or a no-match (nothing returned). `steps` must be ordered by (qid, t), as load_run returns them.
    """
    seen, out = {}, []
    for s in steps:
        titles = seen.setdefault(s["qid"], set())
        if s["kind"] == "Finish":
            out.append("finish_ok" if s["correct"] else "finish_wrong")
        elif s["kind"] == "Search":
            title = s.get("title")
            if s["correct"]:
                out.append("search_ok")
            elif title is None:
                out.append("search_nomatch")
            elif title in titles:
                out.append("search_repeat")
            else:
                out.append("search_distractor")
            if title is not None:
                titles.add(title)
        else:
            out.append("invalid")
    return np.array(out)


def auroc_pairs(score, correct, strata=None):
    """AUROC as the share of (correct, wrong) step pairs ranked correctly, ties counting one half.

    With `strata`, only pairs inside the same stratum count (e.g. equal step index, which removes position from the
    comparison). Without it this equals the ordinary AUROC. Returns nan when no pair exists.
    """
    score, correct = np.asarray(score, float), np.asarray(correct, bool)
    strata = np.zeros(len(score), int) if strata is None else np.asarray(strata)
    win = n = 0.0
    for s in np.unique(strata):
        m = strata == s
        pos, neg = score[m & correct], score[m & ~correct]
        if len(pos) and len(neg):
            r = rankdata(np.concatenate([pos, neg]))
            win += r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
            n += len(pos) * len(neg)
    return win / n if n else float("nan")
