"""Tests for module_order edges in graph generation."""
from __future__ import annotations

from langchain_core.documents import Document
from rag_ed.graphs import graph_from_documents


def test_module_order_edges_directed_and_overwrite():
    # Create three docs in same directory with chronological timestamps but explicit module indices
    docs = [
        Document(page_content="Intro", metadata={"source": "/m/week1/intro.md", "timestamp": "2024-01-01T09:00:00Z", "module_id": "week1", "module_index": 1}),
        Document(page_content="Slides", metadata={"source": "/m/week1/slides.pdf", "timestamp": "2024-01-01T10:00:00Z", "module_id": "week1", "module_index": 2}),
        Document(page_content="Assignment", metadata={"source": "/m/week1/assignment.md", "timestamp": "2024-01-01T11:00:00Z", "module_id": "week1", "module_index": 3}),
    ]

    g = graph_from_documents(docs, prefix="mod")
    G = g.graph
    nodes = sorted(G.nodes)
    # Ensure edges reflect module ordering (1->2, 2->3) with kind=module_order
    # and that these edges exist (they may overwrite chronological ones).
    # Find node ids by matching page content for clarity.
    by_content = {G.nodes[n]['document'].page_content: n for n in nodes}
    intro = by_content['Intro']
    slides = by_content['Slides']
    assign = by_content['Assignment']

    assert G.has_edge(intro, slides)
    assert G.edges[intro, slides].get('kind') == 'module_order'
    assert G.has_edge(slides, assign)
    assert G.edges[slides, assign].get('kind') == 'module_order'

    # Reverse edges may exist due to co_temporal (same-day) linking; ensure they
    # are not tagged as module_order (directional sequencing remains forward).
    if G.has_edge(slides, intro):
        assert G.edges[slides, intro].get('kind') != 'module_order'
    if G.has_edge(assign, slides):
        assert G.edges[assign, slides].get('kind') != 'module_order'


def test_module_order_ignores_non_integer():
    docs = [
        Document(page_content="A", metadata={"source": "/m/w1/a.md", "timestamp": "2024-01-01T09:00:00Z", "module_id": "w1", "module_index": "x"}),
        Document(page_content="B", metadata={"source": "/m/w1/b.md", "timestamp": "2024-01-01T10:00:00Z", "module_id": "w1", "module_index": 2}),
    ]
    g = graph_from_documents(docs, prefix="mod2")
    G = g.graph
    # Only one valid integer index, so no module_order edges should be present.
    # Any edge that exists would be chronological (since they share directory & timestamps ascending) unless module metadata processed.
    edge_kinds = {data.get('kind') for _, _, data in G.edges(data=True)}
    # Should contain chronological but not module_order
    assert 'module_order' not in edge_kinds
