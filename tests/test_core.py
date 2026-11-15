"""Unit tests (no LLM, no internet needed beyond installing deps).  Run:  pytest -q"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import DocuMind  # noqa: E402

SAMPLE = str(ROOT / "data" / "sample.pdf")


@pytest.fixture
def engine():
    return DocuMind(DeterministicFakeEmbedding(size=64), provider="none")


def doc(source, page, text):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_add_pdf_sets_source_and_one_indexed_pages(engine):
    n = engine.add_pdf(SAMPLE, "sample.pdf")
    assert n >= 3
    assert engine.sources == ["sample.pdf"]
    assert {c.metadata["page"] for c in engine.chunks} == {1, 2, 3}


def test_multiple_pdfs_and_source_filter(engine):
    engine.add_pdf(SAMPLE, "a.pdf")
    engine.add_pdf(SAMPLE, "b.pdf")
    assert engine.sources == ["a.pdf", "b.pdf"]
    docs, _ = engine._search("normalization", 3, only=["b.pdf"])
    assert docs and all(d.metadata["source"] == "b.pdf" for d in docs)


def test_rrf_prefers_document_found_by_both_retrievers(engine):
    both, vec_only, kw_only = doc("x.pdf", 1, "both"), doc("x.pdf", 2, "vec"), doc("x.pdf", 3, "kw")
    engine.vs = SimpleNamespace(
        similarity_search_with_score=lambda q, k: [(vec_only, 0.2), (both, 0.4)]
    )
    engine.bm25 = SimpleNamespace(k=0, invoke=lambda q: [kw_only, both])
    docs, conf = engine._search("anything", 3, None)
    assert docs[0] is both                      # appears in both lists -> wins
    assert conf == pytest.approx(1 - 0.2 / 2)   # cosine from squared-L2 distance


def test_guard_refuses_without_calling_llm(engine, monkeypatch):
    monkeypatch.setattr(engine, "_search", lambda *a, **k: ([], 0.05))
    res = engine.ask("capital of France?", history=[])
    assert "couldn't find" in "".join(res.stream)
    assert res.sources == [] and res.label == "Low"


def test_quiz_parses_json_even_when_wrapped_in_text(engine):
    engine.add_pdf(SAMPLE, "sample.pdf")
    reply = (
        'Sure! ```json\n[{"question": "Q?", "options": ["a","b","c","d"], '
        '"answer": 2, "explanation": "because"}]\n```'
    )
    engine.llm = FakeListChatModel(responses=[reply])
    quiz = engine.quiz(n=1)
    assert quiz[0]["answer"] == 2 and len(quiz[0]["options"]) == 4


def test_quiz_raises_on_garbage(engine):
    engine.add_pdf(SAMPLE, "sample.pdf")
    engine.llm = FakeListChatModel(responses=["no json here"])
    with pytest.raises(ValueError):
        engine.quiz(n=1)
