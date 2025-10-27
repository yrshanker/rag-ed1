from langchain_core.documents import Document

import pytest

from rag_ed.graphs import CourseGraph
from rag_ed.retrievers.graph import GraphRetriever


def test_graph_retriever_traverses_neighbors() -> None:
    # Arrange
    graph = CourseGraph()
    graph.add_artifact("a", Document(page_content="A"))
    graph.add_artifact("b", Document(page_content="B"))
    graph.add_artifact("c", Document(page_content="C"))
    graph.add_relationship("a", "b")
    graph.add_relationship("b", "c")
    retriever = GraphRetriever(graph, max_depth=2)

    # Act
    docs = retriever.retrieve("a")

    # Assert
    assert [d.page_content for d in docs] == ["B", "C"]


def test_graph_retriever_raises_for_missing_artifact() -> None:
    # Arrange
    graph = CourseGraph()
    retriever = GraphRetriever(graph)

    # Act / Assert
    with pytest.raises(ValueError, match="Artifact 'x' not found"):
        retriever.retrieve("x")


def test_graph_retriever_raises_for_negative_depth() -> None:
    # Arrange
    graph = CourseGraph()
    graph.add_artifact("a", Document(page_content="A"))
    retriever = GraphRetriever(graph)

    # Act / Assert
    with pytest.raises(ValueError, match="max_depth must be non-negative"):
        retriever.retrieve("a", max_depth=-1)
