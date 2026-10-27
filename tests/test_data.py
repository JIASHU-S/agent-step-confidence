"""TriviaQA evidence-row duplication must not duplicate or overweight source questions."""
import pytest

from agentconf import data


class Rows(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def row(i):
    return {"question_id": f"q{i}", "question": f"Question {i}?",
            "answer": {"value": f"Answer {i}", "aliases": [f"Alias {i}", f"Answer {i}"]}}


def test_trivia_samples_unique_questions_and_is_not_weighted_by_duplicate_rows(monkeypatch):
    base = Rows(row(i) for i in range(20))
    monkeypatch.setattr(data, "_load", lambda *a: base)
    expected = data.load_trivia(8, seed=13)
    expanded = Rows(r for i, r in enumerate(base) for _ in range(i + 1))
    monkeypatch.setattr(data, "_load", lambda *a: expanded)
    actual = data.load_trivia(8, seed=13)
    assert actual == expected
    assert len(actual) == len({r["id"] for r in actual}) == 8
    assert len(data.load_trivia(20, seed=13)) == 20
    assert actual == data.load_trivia(8, seed=13)
    assert actual != data.load_trivia(8, seed=0)
    assert all(len(r["answers"]) == 2 and r["question"].startswith("Question") for r in actual)


def test_trivia_reports_insufficient_unique_questions(monkeypatch):
    monkeypatch.setattr(data, "_load", lambda *a: Rows([row(1)] * 10 + [row(2)] * 10))
    with pytest.raises(ValueError, match="2 unique IDs"):
        data.load_trivia(3)
    assert data.load_trivia(0) == []
