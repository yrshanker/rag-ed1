"""Utilities for constructing course graphs from platform exports."""

from __future__ import annotations

from typing import Iterable

from pathlib import Path
from collections import defaultdict
import re
from itertools import pairwise

import langchain_core.documents

from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader

from .course import CourseGraph


def _safe_date_str(ts: str) -> str | None:
    """Extract YYYY-MM-DD from the beginning of a timestamp string.

    Returns None if not present. Keeps dependencies light and format-agnostic
    (assumes ISO-like timestamps used in tests and loaders).
    """
    if not isinstance(ts, str):
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", ts)
    return m.group(1) if m else None


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

    # Group nodes by parent directory of their source path and by date and stem.
    grouped: dict[Path, list[str]] = defaultdict(list)
    grouped_by_date: dict[str, list[str]] = defaultdict(list)
    grouped_by_stem: dict[str, list[str]] = defaultdict(list)
    for node_id in node_ids:
        doc = graph.graph.nodes[node_id]["document"]
        source = Path(doc.metadata.get("source", "."))
        grouped[source.parent].append(node_id)
        if source.stem:
            grouped_by_stem[source.stem].append(node_id)
        date_str = _safe_date_str(doc.metadata.get("timestamp", ""))
        if date_str:
            grouped_by_date[date_str].append(node_id)

    # Within each directory group, link documents by increasing timestamp.
    for nodes in grouped.values():
        sorted_nodes = sorted(
            nodes,
            key=lambda n: graph.graph.nodes[n]["document"].metadata.get(
                "timestamp", ""
            ),
        )
        for src, dst in pairwise(sorted_nodes):
            # Tag these as chronological to make edge semantics explicit
            graph.add_relationship(src, dst, kind="chronological")

    # Add co-temporal edges between items that share the same day.
    # Add in both directions but do not overwrite existing edge attributes.
    for nodes in grouped_by_date.values():
        if len(nodes) < 2:
            continue
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not graph.graph.has_edge(u, v):
                    graph.add_relationship(u, v, kind="co_temporal")
                if not graph.graph.has_edge(v, u):
                    graph.add_relationship(v, u, kind="co_temporal")

    # Add same-stem edges between items that share the same filename stem
    # (e.g., lecture1.md and lecture1.pdf). Add in both directions without
    # overwriting existing edges.
    for nodes in grouped_by_stem.values():
        if len(nodes) < 2:
            continue
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not graph.graph.has_edge(u, v):
                    graph.add_relationship(u, v, kind="same_stem")
                if not graph.graph.has_edge(v, u):
                    graph.add_relationship(v, u, kind="same_stem")

    # ---------- Canvas module ordering edges ----------
    # If documents contain module metadata (module_id + module_index), connect
    # consecutive items within the same module in ascending order of
    # module_index. These directed edges are tagged kind="module_order" and
    # intentionally overwrite earlier coarse edges (e.g., chronological) to
    # reflect explicit curricular sequencing.
    module_groups: dict[str, list[str]] = defaultdict(list)
    module_index: dict[str, int] = {}
    for node_id in node_ids:
        md = graph.graph.nodes[node_id]["document"].metadata
        mid = md.get("module_id")
        mpos = md.get("module_index")
        if mid is None or mpos is None:
            continue
        try:
            mpos_int = int(mpos)
        except (TypeError, ValueError):  # skip non-integer indices
            continue
        module_groups[mid].append(node_id)
        module_index[node_id] = mpos_int

    for members in module_groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda n: module_index[n])
        for src, dst in pairwise(ordered):
            graph.add_relationship(src, dst, kind="module_order")

    # ---------- Piazza thread edges ----------
    # Detect documents that include Piazza-like metadata keys and add two
    # relationship kinds:
    # - thread_reply: directed parent -> reply
    # - same_thread: hub-and-spoke edges connecting thread root to all members
    by_post_id: dict[str, str] = {}
    per_thread: dict[str, list[str]] = defaultdict(list)
    parent_map: dict[str, str] = {}

    for node_id in node_ids:
        doc = graph.graph.nodes[node_id]["document"]
        md = doc.metadata
        post_id = md.get("post_id")
        thread_id = md.get("thread_id")
        parent_id = md.get("parent_id")
        if post_id:
            by_post_id[post_id] = node_id
        if thread_id and post_id:
            per_thread[thread_id].append(node_id)
        if parent_id and post_id:
            parent_map[node_id] = parent_id

    # thread_reply: parent -> reply. Overwrite any existing edge kind so the
    # conversational parent/child relationship is explicit and takes precedence
    # over coarse chronological adjacency.
    for node, parent_post in parent_map.items():
        parent_node = by_post_id.get(parent_post)
        if parent_node:
            graph.add_relationship(parent_node, node, kind="thread_reply")

    # same_thread: connect thread root (post_id==thread_id or earliest) to others
    for thread_id, members in per_thread.items():
        if len(members) < 2:
            continue
        # prefer the node whose post_id equals the thread_id
        root = None
        for n in members:
            if graph.graph.nodes[n]["document"].metadata.get("post_id") == thread_id:
                root = n
                break
        if root is None:
            # fallback to earliest timestamp
            root = min(
                members,
                key=lambda n: graph.graph.nodes[n]["document"].metadata.get("timestamp", ""),
            )
        for m in members:
            if m == root:
                continue
            # Add same_thread edges without overwriting explicit conversational
            # parent/child relationships: if parent->child was set as
            # 'thread_reply', preserve it.
            if graph.graph.has_edge(root, m):
                existing = graph.graph.get_edge_data(root, m) or {}
                if existing.get("kind") != "thread_reply":
                    graph.add_relationship(root, m, kind="same_thread")
            else:
                graph.add_relationship(root, m, kind="same_thread")

            # m -> root should be same_thread (it's safe to overwrite here).
            graph.add_relationship(m, root, kind="same_thread")

    return graph


def graph_from_documents(
    documents: Iterable[langchain_core.documents.Document], *, prefix: str
) -> CourseGraph:
    """Public helper to build a graph directly from documents.

    This is a thin wrapper around the internal implementation used by
    other graph builders. Exposed primarily for testing and utility scripts.
    """
    return _graph_from_documents(documents, prefix=prefix)


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
