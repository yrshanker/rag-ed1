import types

import langchain_core.documents
from rag_ed.agents import self_querying_retriever_agent as sqra
from rag_ed.agents import vanilla_rag


def test_one_step_retrieval(monkeypatch) -> None:
    class DummyRetriever:
        def __init__(self, *args, **kwargs) -> None:
            self.vector_store = object()

    class DummyQA:
        def run(self, query: str) -> str:  # noqa: D401
            return "dummy"

    def dummy_from_chain_type(*args, **kwargs):
        return DummyQA()

    class DummyOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(vanilla_rag, "VectorStoreRetriever", DummyRetriever)
    monkeypatch.setattr(
        vanilla_rag,
        "RetrievalQA",
        types.SimpleNamespace(from_chain_type=dummy_from_chain_type),
    )
    monkeypatch.setattr(vanilla_rag, "OpenAI", DummyOpenAI)
    assert vanilla_rag.one_step_retrieval("q") == "dummy"


def test_retriever_tool(monkeypatch) -> None:
    class DummyRetriever:
        def retrieve(
            self, query: str, k: int
        ) -> list[langchain_core.documents.Document]:
            return [langchain_core.documents.Document(page_content="d")]

    monkeypatch.setattr(sqra, "VectorStoreRetriever", lambda *a, **k: DummyRetriever())
    tool = sqra.create_retriever_tool("c", "p")
    assert "d" in tool.forward("anything")


def test_create_agent(monkeypatch) -> None:
    class DummyCodeAgent:
        def __init__(
            self, tools, model, max_steps, verbosity_level
        ) -> None:  # noqa: D401
            self.tools = tools
            self.model = model

    class DummyModel:
        pass

    monkeypatch.setattr(sqra, "VectorStoreRetriever", lambda *a, **k: object())
    monkeypatch.setattr(sqra, "CodeAgent", DummyCodeAgent)
    monkeypatch.setattr(sqra, "OpenAIServerModel", DummyModel)
    agent = sqra.create_agent("c", "p")
    assert len(agent.tools) == 1
