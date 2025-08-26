"""Tests for agent modules."""

import types

import langchain_core.documents

from rag_ed.agents import self_querying, vanilla_rag


def test_one_step_retrieval(monkeypatch) -> None:
    """``one_step_retrieval`` delegates to the underlying QA chain."""

    class DummyRetriever:
        def __init__(self, *args, **kwargs) -> None:
            self.vector_store = object()

    class DummyQA:
        def run(self, query: str) -> str:  # noqa: D401
            return "dummy"

    def dummy_from_chain_type(*args, **kwargs):
        return DummyQA()

    class DummyOpenAI:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

    monkeypatch.setattr(vanilla_rag, "VectorStoreRetriever", DummyRetriever)
    monkeypatch.setattr(
        vanilla_rag,
        "RetrievalQA",
        types.SimpleNamespace(from_chain_type=dummy_from_chain_type),
    )
    monkeypatch.setattr(vanilla_rag, "OpenAI", DummyOpenAI)
    assert (
        vanilla_rag.one_step_retrieval("q", canvas_path="c", piazza_path="p") == "dummy"
    )


def test_run_agent_decomposes_queries(monkeypatch) -> None:
    """The self-querying agent retrieves for each sub-query."""

    class DummyRetriever:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            self.queries: list[str] = []

        def retrieve(
            self, query: str, k: int
        ) -> list[langchain_core.documents.Document]:  # noqa: D401
            self.queries.append(query)
            return [
                langchain_core.documents.Document(page_content=f"result for {query}")
            ]

    dummy = DummyRetriever()
    monkeypatch.setenv("CANVAS_PATH", "c")
    monkeypatch.setenv("PIAZZA_PATH", "p")
    monkeypatch.setattr(self_querying, "VectorStoreRetriever", lambda *a, **k: dummy)
    self_querying._RETRIEVER = None

    result = self_querying.run_agent("first part and then second part")

    assert dummy.queries == ["first part", "second part"]
    assert "result for first part" in result
    assert "result for second part" in result
