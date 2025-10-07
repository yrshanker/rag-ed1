"""Tests for CombinedRetriever (fusion of vector + graph)."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag_ed.retrievers.graph import GraphRetriever
from rag_ed.retrievers.fusion import CombinedRetriever, FusionConfig
from rag_ed.graphs import CourseGraph


class DummyVectorRetriever(BaseRetriever):
    def __init__(self, docs):
        self._docs = docs
    def _get_relevant_documents(self, query: str, *, run_manager):  # pragma: no cover - trivial
        # Return docs in provided order to simulate ranking
        return self._docs
    def retrieve(self, query: str):  # pragma: no cover - simple pass-through
        return self._docs


def build_graph_with_edges():
    g = CourseGraph()
    # root node id matches query we will use
    g.add_artifact("root", Document(page_content="ROOT", metadata={"source": "/root.md"}))
    g.add_artifact("a", Document(page_content="A", metadata={"source": "/a.md"}))
    g.add_artifact("b", Document(page_content="B", metadata={"source": "/b.md"}))
    # root -> a (same_stem weight 5), root -> b (chronological weight 1)
    g.add_relationship("root", "a", kind="same_stem")
    g.add_relationship("root", "b", kind="chronological")
    return g


def test_fusion_priority_with_weights():
    vector_docs = [
        Document(page_content="Vec1", metadata={"source": "/v1.md"}),
        Document(page_content="Vec2", metadata={"source": "/v2.md"}),
    ]
    vec = DummyVectorRetriever(vector_docs)
    graph = GraphRetriever(build_graph_with_edges(), max_depth=1, edge_weights={"same_stem": 5.0, "chronological": 1.0})
    fusion = CombinedRetriever(vector_retriever=vec, graph_retriever=graph, config=FusionConfig(alpha=1.0, beta=1.0, k=4))
    results = fusion.retrieve("root")
    # Scores (approx reasoning):
    # graph same_stem (a): weight 5
    # graph chronological (b): weight 1
    # vector ranks: Vec1 -> 1/(1+0)=1.0; Vec2 -> 1/(1+1)=0.5
    # So ordering should start with A (5), then Vec1 (1.0), then B (1), then Vec2 (0.5) -> A, Vec1, B, Vec2
    contents = [d.page_content for d in results]
    assert contents == ["A", "Vec1", "B", "Vec2"]


def test_fusion_tie_breaks_stable():
    # All vector docs only; no graph contribution -> ordering stays vector order due to score then source fallback.
    vector_docs = [
        Document(page_content="Vec1", metadata={"source": "/v1.md"}),
        Document(page_content="Vec2", metadata={"source": "/v2.md"}),
    ]
    vec = DummyVectorRetriever(vector_docs)
    # Graph with no neighbors for query ensures no graph scores
    g = CourseGraph()
    g.add_artifact("root", Document(page_content="ROOT", metadata={"source": "/root.md"}))
    graph = GraphRetriever(g, max_depth=1)
    fusion = CombinedRetriever(vector_retriever=vec, graph_retriever=graph, config=FusionConfig(alpha=1.0, beta=1.0, k=2))
    results = fusion.retrieve("root")
    contents = [d.page_content for d in results]
    assert contents == ["Vec1", "Vec2"]
