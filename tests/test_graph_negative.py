import pytest
from rag_ed.graphs.course import CourseGraph
from rag_ed.retrievers.graph import GraphRetriever
from langchain_core.documents import Document


def test_course_graph_neighbors_keyerror():
    graph = CourseGraph()
    graph.add_artifact("a", Document(page_content="A"))
    with pytest.raises(KeyError, match="Artifact ID 'missing' not found in graph."):
        graph.neighbors("missing")


def test_graph_retriever_keyerror():
    graph = CourseGraph()
    graph.add_artifact("a", Document(page_content="A"))
    retriever = GraphRetriever(graph)
    with pytest.raises(KeyError, match="Artifact ID 'missing' not found in graph."):
        retriever.retrieve("missing")
