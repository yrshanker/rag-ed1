"""Integration test: ensure PiazzaLoader yields thread metadata that the graph can use."""
from __future__ import annotations

from tests.piazza_utils import generate_piazza_export
from rag_ed.graphs import graph_from_piazza


def test_piazza_loader_thread_integration(tmp_path):
    piazza_file = tmp_path / "piazza.zip"
    generate_piazza_export(piazza_file, num_posts=3)

    graph = graph_from_piazza(str(piazza_file))
    G = graph.graph

    # collect nodes by post_id from metadata
    post_map = {}
    for n, data in G.nodes(data=True):
        pid = data["document"].metadata.get("post_id")
        if pid:
            post_map[pid] = n

    # We expect p1->p2 (thread_reply) and p2->p3 (thread_reply)
    assert "p1" in post_map and "p2" in post_map and "p3" in post_map
    assert G.has_edge(post_map["p1"], post_map["p2"])
    assert G.get_edge_data(post_map["p1"], post_map["p2"]).get("kind") == "thread_reply"
    assert G.has_edge(post_map["p2"], post_map["p3"])
    assert G.get_edge_data(post_map["p2"], post_map["p3"]).get("kind") == "thread_reply"
