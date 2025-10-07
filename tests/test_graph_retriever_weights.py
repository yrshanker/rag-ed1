"""Tests for GraphRetriever edge weighting behavior."""
from __future__ import annotations

from langchain_core.documents import Document
from rag_ed.graphs import graph_from_documents
from rag_ed.retrievers.graph import GraphRetriever


def test_graph_retriever_weights_prioritizes_higher_kind():
    # Create three documents forming edges of different kinds from root A
    docs = [
        Document(page_content="Root", metadata={"source": "/s/a.md", "timestamp": "2024-01-01T00:00:00Z"}),
        Document(page_content="Chrono", metadata={"source": "/s/b.md", "timestamp": "2024-01-01T01:00:00Z"}),
        Document(page_content="SameStem", metadata={"source": "/t/a.pdf", "timestamp": "2024-01-02T00:00:00Z"}),
    ]
    g = graph_from_documents(docs, prefix="w")
    G = g.graph
    # Identify nodes
    by_content = {G.nodes[n]['document'].page_content: n for n in G.nodes}
    root = by_content['Root']

    # Sanity: ensure both chronological and same_stem edges exist out of root
    kinds_from_root = {G.get_edge_data(root, nbr).get('kind') for nbr in G.successors(root)}
    assert 'chronological' in kinds_from_root
    # same_stem edge may appear in either direction; ensure at least one neighbor has kind same_stem
    has_same_stem = any(
        (G.get_edge_data(root, nbr) or {}).get('kind') == 'same_stem' or (G.get_edge_data(nbr, root) or {}).get('kind') == 'same_stem'
        for nbr in G.nodes if nbr != root
    )
    assert has_same_stem

    retriever = GraphRetriever(g, max_depth=1, edge_weights={'same_stem': 5.0, 'chronological': 1.0})
    docs_out = retriever.retrieve(root)
    contents = [d.page_content for d in docs_out]
    # Expect SameStem before Chrono due to higher weight
    assert contents == sorted(contents, key=lambda c: 0 if c == 'SameStem' else 1)
    assert contents[0] == 'SameStem'


def test_graph_retriever_weights_deterministic_ties():
    # Two chronological edges (simulate by making two later docs in same directory)
    docs = [
        Document(page_content="A", metadata={"source": "/d/a.md", "timestamp": "2024-01-01T00:00:00Z"}),
        Document(page_content="B", metadata={"source": "/d/b.md", "timestamp": "2024-01-01T01:00:00Z"}),
        Document(page_content="C", metadata={"source": "/d/c.md", "timestamp": "2024-01-01T02:00:00Z"}),
    ]
    g = graph_from_documents(docs, prefix="w2")
    G = g.graph
    by_content = {G.nodes[n]['document'].page_content: n for n in G.nodes}
    a = by_content['A']

    retriever = GraphRetriever(g, max_depth=2, edge_weights={'chronological': 1.0})
    docs_out = retriever.retrieve(a)
    # With equal weights, ordering should be deterministic by node id
    ids = [node_id for _, node_id in sorted(((G.nodes[n]['document'].page_content, n) for n in by_content.values()))]
    # Extract returned contents
    contents = [d.page_content for d in docs_out]
    assert set(contents) == {'B', 'C'}
    # Determinism: since both chronological with same weight, they should appear in node-id order
    # Node id order is stable by creation sequence: w2_1 then w2_2
    assert contents == ['B', 'C']
