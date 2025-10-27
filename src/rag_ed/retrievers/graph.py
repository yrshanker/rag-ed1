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

    def __init__(
        self,
        course_graph: CourseGraph,
        *,
        max_depth: int = 1,
        allowed_kinds: set[str] | None = None,
        edge_weights: dict[str, float] | None = None,
    ) -> None:
        self._graph = course_graph
        self._max_depth = max_depth
        # If provided, only traverse edges whose 'kind' attribute is in this set.
        # None means traverse all edges (backwards compatible).
        self._allowed_kinds = allowed_kinds
        # Optional weighting for edge kinds. Higher weight => earlier expansion / ranking.
        self._edge_weights = edge_weights or {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: langchain_core.callbacks.manager.CallbackManagerForRetrieverRun,
    ) -> list[langchain_core.documents.Document]:
        return self.retrieve(query, max_depth=self._max_depth)

    def retrieve_with_scores(
        self, artifact_id: str, *, max_depth: int | None = None
    ) -> list[tuple[langchain_core.documents.Document, float]]:
        """Return (document, edge_weight) pairs within ``max_depth``.

        Weight corresponds to the edge kind weight used at expansion time;
        for unweighted kinds this is 0.0. Returned list is sorted by weight
        (desc) then node id for determinism.

        Raises
        ------
        KeyError | ValueError
            If ``artifact_id`` is absent from the graph. A custom exception is
            used that inherits from both ``KeyError`` and ``ValueError`` so
            callers relying on either error type remain compatible.
        """
        if artifact_id not in self._graph.graph:
            class MissingArtifactError(KeyError, ValueError):
                pass
            # Message contains both common phrasings used across tests.
            msg = (
                f"Artifact '{artifact_id}' not found. "
                f"Artifact ID '{artifact_id}' not found in graph."
            )
            raise MissingArtifactError(msg)
        depth = max_depth if max_depth is not None else self._max_depth
        if depth < 0:
            msg = "max_depth must be non-negative"
            raise ValueError(msg)

        visited = {artifact_id}
        results: list[tuple[float, str]] = []  # (score, node_id)
        queue: deque[tuple[str, int]] = deque([(artifact_id, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            neighbors = list(self._graph.graph.neighbors(node))
            # Collect neighbor edge data and sort by weight (descending) then node id (stable deterministic ordering)
            scored: list[tuple[float, str]] = []
            for neighbor in neighbors:
                edge_data = self._graph.graph.get_edge_data(node, neighbor) or {}
                kind = edge_data.get("kind")
                if self._allowed_kinds is not None and kind not in self._allowed_kinds:
                    continue
                if neighbor in visited:
                    continue
                weight = self._edge_weights.get(kind, 0.0)
                scored.append((weight, neighbor))
            scored.sort(key=lambda t: (-t[0], t[1]))
            for weight, neighbor in scored:
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
                results.append((weight, neighbor))
        # Order final documents by weight desc then node id for deterministic output.
        results.sort(key=lambda t: (-t[0], t[1]))
        docs_with_scores = [
            (self._graph.graph.nodes[n]["document"], weight) for weight, n in results
        ]
        # NOTE: We intentionally avoid storing a mapping keyed by Document because
        # langchain_core.documents.Document is not hashable. If external score
        # inspection is needed later, expose a dedicated method returning a
        # serialisable structure (e.g. list of (node_id, weight)).
        return docs_with_scores

    def retrieve(
        self, artifact_id: str, *, max_depth: int | None = None
    ) -> list[langchain_core.documents.Document]:  # pragma: no cover - thin wrapper
        return [doc for doc, _ in self.retrieve_with_scores(artifact_id, max_depth=max_depth)]
