# Step-label audit: annotation guidelines (v1)

You will judge steps taken by a question-answering agent. The agent answers a two-hop question by calling
`Search[query]`, which returns one paragraph from a fixed set of ten, and `Finish[answer]`, which ends the episode.
Each row of `annotate_steps.csv` is one step. Work on your own copy and do not discuss items with the other
annotator until both of you have finished.

## What you see

| column | meaning |
| --- | --- |
| `question`, `gold_answer` | the question and the dataset's reference answer |
| `history` | earlier actions in the same episode and the title each search returned |
| `step_text` | the agent's thought and action for this step |
| `action`, `argument` | `Search`, `Finish` or `invalid`, and its argument |
| `returned_title`, `returned_text` | the paragraph this search returned, as the agent saw it (Search only) |

You are **not** shown the automatic label, the model, or which paragraphs the dataset marks as supporting. Judge
from the question, the gold answer and the text.

## Codes (write exactly one in `code`)

**Search steps.** Judge whether this search moved the agent closer to the answer.

| code | use when |
| --- | --- |
| `S-USEFUL` | the returned paragraph is new (not in `history`) and contains a fact needed to answer the question |
| `S-PARTIAL` | the paragraph is new and relevant (for example it names the bridge entity) but is not one of the facts needed |
| `S-UNHELPFUL` | the paragraph is new but does not help answer the question |
| `S-REPEAT` | the returned title already appears in `history` |
| `UNCLEAR` | you cannot decide; explain in `notes` |

**Finish steps.** Judge the answer against the question, using the gold answer as a guide.

| code | use when |
| --- | --- |
| `F-CORRECT` | the answer means the same as the gold answer (aliases, spelling, extra harmless words are fine) |
| `F-PARTIAL` | partly right: too broad, too narrow, or missing part of a list |
| `F-WRONG` | a different answer, or no answer |
| `F-GOLD-ISSUE` | the gold answer looks wrong or the question is ambiguous; explain in `notes` |
| `UNCLEAR` | you cannot decide; explain in `notes` |

**Invalid steps** (the action could not be parsed).

| code | use when |
| --- | --- |
| `I-INVALID` | there is no usable action |
| `I-VALID` | a reasonable reader would recognise a valid Search or Finish action; explain in `notes` |

## Episodes (`annotate_episodes.csv`)

Each row is a full episode transcript and its final answer. Code `E-CORRECT`, `E-PARTIAL`, `E-WRONG`,
`E-GOLD-ISSUE` or `UNCLEAR` with the same meanings as for Finish steps. An episode with no final answer is
`E-WRONG`. Steps of these episodes also appear in the step sheet (`episode_item` names the episode); code them
there.

## Practicalities

- Spend about one minute per step; do not search the web.
- Use `notes` freely, especially for `UNCLEAR`, `F-GOLD-ISSUE` and `I-VALID`.
- Save as CSV (UTF-8). Do not reorder, delete or add rows or columns.
