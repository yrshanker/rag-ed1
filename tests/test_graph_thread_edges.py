"""Unit tests for Piazza thread-based graph edges (synthetic)."""
from __future__ import annotations

from langchain_core.documents import Document
from rag_ed.graphs import graph_from_documents


def test_piazza_thread_edges():
    # Create synthetic posts: root -> reply1 -> reply2
    docs = [
        Document(
            page_content="root",
            metadata={
                "source": "/piazza/p1.json",
                "timestamp": "2025-09-01T09:00:00Z",
                "post_id": "p1",
                "thread_id": "p1",
            },
        ),
        Document(
            page_content="reply1",
            metadata={
                "source": "/piazza/p2.json",
                "timestamp": "2025-09-01T10:00:00Z",
                "post_id": "p2",
                "thread_id": "p1",
                "parent_id": "p1",
            },
        ),
        Document(
            page_content="reply2",
            metadata={
                "source": "/piazza/p3.json",
                "timestamp": "2025-09-01T11:00:00Z",
                "post_id": "p3",
                "thread_id": "p1",
                "parent_id": "p2",
            },
        ),
    ]

    graph = graph_from_documents(docs, prefix="piazza")
    G = graph.graph
    nodes = list(G.nodes)
    # Map post_id -> node id
    post_map = {}
    for n in nodes:
        post = G.nodes[n]["document"].metadata.get("post_id")
        if post:
            post_map[post] = n

    # Assertions:
    # thread_reply: p1 -> p2, p2 -> p3
    assert G.has_edge(post_map["p1"], post_map["p2"])
    assert G.get_edge_data(post_map["p1"], post_map["p2"]).get("kind") == "thread_reply"
    assert G.has_edge(post_map["p2"], post_map["p3"])
    assert G.get_edge_data(post_map["p2"], post_map["p3"]).get("kind") == "thread_reply"

    # same_thread: root should connect bidirectionally to all members
    assert G.has_edge(post_map["p1"], post_map["p2"]) and G.has_edge(post_map["p2"], post_map["p1"])
    assert G.get_edge_data(post_map["p1"], post_map["p2"]).get("kind") in {"thread_reply", "same_thread"}
    assert G.get_edge_data(post_map["p2"], post_map["p1"]).get("kind") == "same_thread"

    assert G.has_edge(post_map["p1"], post_map["p3"]) and G.has_edge(post_map["p3"], post_map["p1"])
    assert G.get_edge_data(post_map["p1"], post_map["p3"]).get("kind") == "same_thread"
    assert G.get_edge_data(post_map["p3"], post_map["p1"]).get("kind") == "same_thread"
