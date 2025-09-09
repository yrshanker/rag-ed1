"""Retrieve documents by traversing a course graph."""

from __future__ import annotations

from collections import deque

import langchain_core.callbacks.manager
import langchain_core.documents
import langchain_core.retrievers

from rag_ed.graphs import CourseGraph


class GraphRetriever(langchain_core.retrievers.BaseRetriever):
    """Traverse a :class:`~rag_ed.graphs.CourseGraph` to fetch related documents.

    Parameters
    ----------
    course_graph : CourseGraph
        Graph containing course artifacts.
    max_depth : int, optional
        Maximum traversal depth. Defaults to ``1``.

    Examples
    --------
    >>> from langchain_core.documents import Document
    >>> from rag_ed.graphs import CourseGraph
    >>> graph = CourseGraph()
    >>> graph.add_artifact("a", Document(page_content="A"))
    >>> graph.add_artifact("b", Document(page_content="B"))
    >>> graph.add_relationship("a", "b")
    >>> retriever = GraphRetriever(graph, max_depth=1)
    >>> [d.page_content for d in retriever.retrieve("a")]
    ['B']
    """

    def __init__(self, course_graph: CourseGraph, *, max_depth: int = 1) -> None:
        self._graph = course_graph
        self._max_depth = max_depth

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: langchain_core.callbacks.manager.CallbackManagerForRetrieverRun,
    ) -> list[langchain_core.documents.Document]:
        return self.retrieve(query, max_depth=self._max_depth)

    def retrieve(
        self, artifact_id: str, *, max_depth: int | None = None
    ) -> list[langchain_core.documents.Document]:
        """Return documents connected to ``artifact_id`` within ``max_depth``.

        Raises
        ------
        KeyError
            If ``artifact_id`` is not present in the graph.
        """
        if artifact_id not in self._graph.graph:
            raise KeyError(f"Artifact ID '{artifact_id}' not found in graph.")
        depth = max_depth if max_depth is not None else self._max_depth
        visited = {artifact_id}
        docs: list[langchain_core.documents.Document] = []
        queue: deque[tuple[str, int]] = deque([(artifact_id, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for neighbor in self._graph.graph.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
                docs.append(self._graph.graph.nodes[neighbor]["document"])
        return docs
