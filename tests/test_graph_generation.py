from pathlib import Path

from rag_ed.graphs import graph_from_canvas, graph_from_piazza
from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export
from langchain_core.documents import Document
from rag_ed.graphs import graph_from_documents


def test_graph_from_canvas(tmp_path: Path) -> None:
    # Arrange
    imscc_path = generate_imscc(tmp_path / "course")

    # Act
    graph = graph_from_canvas(str(imscc_path))

    # Assert
    assert len(graph.graph.nodes) >= 3
    web_nodes = [
        n
        for n, data in graph.graph.nodes(data=True)
        if "webcontent" in data["document"].metadata.get("source", "")
    ]
    assert len(web_nodes) == 2
    # Nodes should be linked chronologically within the webcontent directory.
    sorted_web = sorted(
        web_nodes,
        key=lambda n: graph.graph.nodes[n]["document"].metadata["timestamp"],
    )
    assert graph.graph.has_edge(sorted_web[0], sorted_web[1])


def test_graph_from_piazza(tmp_path: Path) -> None:
    # Arrange
    piazza_path = generate_piazza_export(tmp_path / "piazza")

    # Act
    graph = graph_from_piazza(str(piazza_path))

    # Assert
    assert len(graph.graph.nodes) >= 3
    root_nodes = list(graph.graph.nodes)
    sorted_root = sorted(
        root_nodes,
        key=lambda n: graph.graph.nodes[n]["document"].metadata["timestamp"],
    )
    assert graph.graph.has_edge(sorted_root[0], sorted_root[1])
    assert graph.graph.has_edge(sorted_root[1], sorted_root[2])


def test_generated_edges_are_tagged_chronological() -> None:
    # Arrange: two docs in same directory with increasing timestamps
    docs = [
        Document(
            page_content="A",
            metadata={
                "source": "/course/week1/a.md",
                "timestamp": "2024-09-01T09:00:00Z",
            },
        ),
        Document(
            page_content="B",
            metadata={
                "source": "/course/week1/b.md",
                "timestamp": "2024-09-01T10:00:00Z",
            },
        ),
    ]

    # Act
    graph = graph_from_documents(docs, prefix="t")

    # Assert
    G = graph.graph
    nodes = sorted(G.nodes)
    assert len(nodes) == 2
    u, v = nodes[0], nodes[1]
    assert G.has_edge(u, v)
    assert G.get_edge_data(u, v).get("kind") == "chronological"


def test_co_temporal_edges_bidirectional() -> None:
    # Arrange: two docs with the same day in different source directories
    docs = [
        Document(
            page_content="X",
            metadata={
                "source": "/course/week1/x.md",
                "timestamp": "2024-09-05T08:00:00Z",
            },
        ),
        Document(
            page_content="Y",
            metadata={
                "source": "/course/week2/y.md",
                "timestamp": "2024-09-05T12:00:00Z",
            },
        ),
        Document(
            page_content="Z",
            metadata={
                "source": "/course/week3/z.md",
                "timestamp": "2024-09-06T12:00:00Z",
            },
        ),
    ]

    # Act
    graph = graph_from_documents(docs, prefix="t")

    # Assert: nodes with the same date should connect in both directions
    G = graph.graph
    nodes = sorted(G.nodes)
    n0, n1, n2 = nodes
    assert G.has_edge(n0, n1)
    assert G.has_edge(n1, n0)
    kinds = {
        G.get_edge_data(n0, n1).get("kind"),
        G.get_edge_data(n1, n0).get("kind"),
    }
    assert "co_temporal" in kinds


def test_same_stem_edges_bidirectional() -> None:
    # Arrange: two docs with the same stem but different extensions/dirs
    docs = [
        Document(
            page_content="L1 text",
            metadata={
                "source": "/course/week1/lecture1.md",
                "timestamp": "2024-09-07T09:00:00Z",
            },
        ),
        Document(
            page_content="L1 pdf",
            metadata={
                "source": "/course/week2/lecture1.pdf",
                "timestamp": "2024-09-08T09:00:00Z",
            },
        ),
        Document(
            page_content="L2 text",
            metadata={
                "source": "/course/week1/lecture2.md",
                "timestamp": "2024-09-07T10:00:00Z",
            },
        ),
    ]

    # Act
    graph = graph_from_documents(docs, prefix="t")

    # Assert: nodes with the same stem should connect both ways with same_stem
    G = graph.graph
    # Find nodes grouped by stem
    stems = {}
    for n, data in G.nodes(data=True):
        stem = Path(data["document"].metadata["source"]).stem
        stems.setdefault(stem, []).append(n)
    l1_nodes = stems["lecture1"]
    assert len(l1_nodes) == 2
    u, v = l1_nodes
    assert G.has_edge(u, v)
    assert G.has_edge(v, u)
    assert G.get_edge_data(u, v).get("kind") == "same_stem"
    assert G.get_edge_data(v, u).get("kind") == "same_stem"
