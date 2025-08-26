"""Utilities for constructing course graphs from platform exports."""

from __future__ import annotations

from typing import Iterable

from pathlib import Path
from collections import defaultdict
from itertools import pairwise

import langchain_core.documents

from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader

from .course import CourseGraph


def _graph_from_documents(
    documents: Iterable[langchain_core.documents.Document], *, prefix: str
) -> CourseGraph:
    """Build a :class:`CourseGraph` from an iterable of documents.

    Nodes are connected more intelligently than simple sequential linking. The
    loader groups documents by their source directory (from ``metadata['source']``)
    and links them in chronological order using the ``timestamp`` metadata. This
    preserves basic structural and temporal relationships among related
    artifacts.

    Parameters
    ----------
    documents : Iterable[Document]
        Documents to add as graph nodes.
    prefix : str
        Prefix used when generating node identifiers.

    Returns
    -------
    CourseGraph
        Graph containing all ``documents`` where edges reflect directory
        groupings and timestamp ordering.
    """
    graph = CourseGraph()
    node_ids: list[str] = []
    for idx, doc in enumerate(documents):
        node_id = f"{prefix}_{idx}"
        graph.add_artifact(node_id, doc)
        node_ids.append(node_id)

    # Group nodes by parent directory of their source path.
    grouped: dict[Path, list[str]] = defaultdict(list)
    for node_id in node_ids:
        doc = graph.graph.nodes[node_id]["document"]
        source = Path(doc.metadata.get("source", "."))
        grouped[source.parent].append(node_id)

    # Within each directory group, link documents by increasing timestamp.
    for nodes in grouped.values():
        sorted_nodes = sorted(
            nodes,
            key=lambda n: graph.graph.nodes[n]["document"].metadata.get(
                "timestamp", ""
            ),
        )
        for src, dst in pairwise(sorted_nodes):
            graph.add_relationship(src, dst)

    return graph


def graph_from_canvas(canvas_path: str) -> CourseGraph:
    """Create a graph from a Canvas export.

    Examples
    --------
    >>> from rag_ed.graphs import graph_from_canvas
    >>> graph = graph_from_canvas("/path/to/course.imscc")
    >>> list(graph.graph.nodes)  # doctest: +SKIP
    ['canvas_0', 'canvas_1']
    """
    documents = CanvasLoader(canvas_path).load()
    return _graph_from_documents(documents, prefix="canvas")


def graph_from_piazza(piazza_path: str) -> CourseGraph:
    """Create a graph from a Piazza export.

    Examples
    --------
    >>> from rag_ed.graphs import graph_from_piazza
    >>> graph = graph_from_piazza("/path/to/piazza.zip")
    >>> list(graph.graph.nodes)  # doctest: +SKIP
    ['piazza_0', 'piazza_1', 'piazza_2']
    """
    documents = PiazzaLoader(piazza_path).load()
    return _graph_from_documents(documents, prefix="piazza")
