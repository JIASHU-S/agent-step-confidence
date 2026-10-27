"""Prompt templates. The ReAct few-shot examples are hand-written and not taken from HotpotQA."""

REACT_SYSTEM = """Solve a question answering task by interleaving Thought, Action and Observation steps.
Thought reasons about the current situation. Action is one of:
(1) Search[entity or query]: returns the most relevant paragraph from the documents listed for this question.
(2) Finish[answer]: returns a short answer and finishes the task.
Write exactly one Thought and one Action per step, then stop and wait for the Observation.

Example 1
Documents: Spirited Away; Hayao Miyazaki; Studio Ghibli
Question: What is the nationality of the director of the film Spirited Away?
Thought 1: I need to find who directed Spirited Away.
Action 1: Search[Spirited Away]
Observation 1: [Spirited Away] Spirited Away is a 2001 Japanese animated fantasy film written and directed by Hayao Miyazaki.
Thought 2: The director is Hayao Miyazaki. I need his nationality.
Action 2: Search[Hayao Miyazaki]
Observation 2: [Hayao Miyazaki] Hayao Miyazaki is a Japanese animator, filmmaker and manga artist.
Thought 3: Hayao Miyazaki is Japanese.
Action 3: Finish[Japanese]

Example 2
Documents: Kyoto University; Tohoku University; Osaka University
Question: Which was founded first, Kyoto University or Tohoku University?
Thought 1: I need the founding years of both universities.
Action 1: Search[Kyoto University]
Observation 1: [Kyoto University] Kyoto University is a national research university in Kyoto, Japan, founded in 1897.
Thought 2: Kyoto University was founded in 1897. Now Tohoku University.
Action 2: Search[Tohoku University]
Observation 2: [Tohoku University] Tohoku University is a public research university in Sendai, Japan, established in 1907.
Thought 3: 1897 is earlier than 1907, so Kyoto University was founded first.
Action 3: Finish[Kyoto University]"""


def react_messages(example):
    user = f"Documents: {'; '.join(example['titles'])}\nQuestion: {example['question']}"
    return [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": user}]


VERBAL_SYSTEM = "You review the steps of a question-answering agent that searches a small set of documents."


def verbal_messages(example, trajectory_text, step_idx):
    user = (
        f"Documents: {'; '.join(example['titles'])}\n"
        f"Question: {example['question']}\n\n"
        f"Agent trajectory so far:\n{trajectory_text.strip()}\n\n"
        f"How confident are you that Action {step_idx} is correct, meaning it retrieves a document that is needed "
        f"to answer the question and was not retrieved before, or it gives the correct final answer? "
        f"Reply with a single integer from 0 to 100."
    )
    return [{"role": "system", "content": VERBAL_SYSTEM}, {"role": "user", "content": user}]


QA_SYSTEM = "You answer trivia questions. Reply with the answer only, as a short phrase, with no explanation."
QA_FEW_SHOT = [
    ("What is the capital city of Australia?", "Canberra"),
    ("Who wrote the novel 'Pride and Prejudice'?", "Jane Austen"),
    ("Which element has the chemical symbol Fe?", "Iron"),
]


def qa_messages(question):
    msgs = [{"role": "system", "content": QA_SYSTEM}]
    for q, a in QA_FEW_SHOT:
        msgs += [{"role": "user", "content": f"Question: {q}"}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs


def render(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Plain-text fallback for base models / tests.
    return "".join(m["content"] + "\n" for m in messages) + "\n"
