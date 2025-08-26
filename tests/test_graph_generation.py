from pathlib import Path

from rag_ed.graphs import graph_from_canvas, graph_from_piazza
from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export


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
