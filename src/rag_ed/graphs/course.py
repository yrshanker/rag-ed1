"""Course artifact graph modeled with ``networkx``."""

from __future__ import annotations

import networkx as nx
import langchain_core.documents


class CourseGraph:
    """Graph representing relationships among course artifacts.

    Nodes store :class:`langchain_core.documents.Document` instances.
    Edges indicate relationships like references or sequencing.

    Examples
    --------
    >>> from langchain_core.documents import Document
    >>> from rag_ed.graphs import CourseGraph
    >>> graph = CourseGraph()
    >>> graph.add_artifact("a", Document(page_content="A"))
    >>> graph.add_artifact("b", Document(page_content="B"))
    >>> graph.add_relationship("a", "b")
    >>> [d.page_content for d in graph.neighbors("a")]
    ['B']
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_artifact(
        self, artifact_id: str, document: langchain_core.documents.Document
    ) -> None:
        """Add a document to the graph."""
        self._graph.add_node(artifact_id, document=document)

    def add_relationship(self, source_id: str, target_id: str) -> None:
        """Create a directed edge between two artifacts."""
        self._graph.add_edge(source_id, target_id)

    def neighbors(self, artifact_id: str) -> list[langchain_core.documents.Document]:
        """Return documents directly connected to ``artifact_id``.

        Raises
        ------
        KeyError
            If ``artifact_id`` is not present in the graph.
        """
        if artifact_id not in self._graph:
            raise KeyError(f"Artifact ID '{artifact_id}' not found in graph.")
        return [
            self._graph.nodes[n]["document"] for n in self._graph.neighbors(artifact_id)
        ]

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying ``networkx`` graph."""
        return self._graph
