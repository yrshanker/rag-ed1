"""Tests for GraphRetriever.allowed_kinds filtering behavior."""
from __future__ import annotations

from langchain_core.documents import Document
from rag_ed.graphs import graph_from_documents
from rag_ed.retrievers.graph import GraphRetriever


def test_graph_retriever_allowed_kinds():
    # Create three documents: A -> B (chronological), A -> C (same_stem)
    docs = [
        Document(page_content="A", metadata={"source": "/c/a.md", "timestamp": "2024-01-01T00:00:00Z"}),
        Document(page_content="B", metadata={"source": "/c/b.md", "timestamp": "2024-01-01T01:00:00Z"}),
        Document(page_content="C", metadata={"source": "/d/a.pdf", "timestamp": "2024-01-02T00:00:00Z"}),
    ]

    graph = graph_from_documents(docs, prefix="t")
    # Ensure both edge kinds exist
    G = graph.graph
    nodes = sorted(G.nodes)
    a, b, c = nodes
    # Note: nodes are sorted to obtain deterministic IDs (stable across runs)
    # a->b chronological should exist, a->c same_stem should exist via stem 'a'
    assert G.has_edge(a, b)
    assert any(data.get("kind") == "chronological" for _, _, data in G.edges(data=True))
    assert any(data.get("kind") == "same_stem" for _, _, data in G.edges(data=True))

    # Default retriever (no filter) returns both neighbors
    retriever_all = GraphRetriever(graph, max_depth=1)
    res_all = retriever_all.retrieve(a)
    assert len(res_all) == 2

    # Filter only same_stem
    retriever_same = GraphRetriever(graph, max_depth=1, allowed_kinds={"same_stem"})
    res_same = retriever_same.retrieve(a)
    assert len(res_same) == 1
    assert res_same[0].page_content == "C"