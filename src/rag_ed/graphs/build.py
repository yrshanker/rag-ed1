"""Utilities to build a combined CourseGraph from multiple sources.

Current sources: Canvas IMSCC archive + Piazza ZIP export.

This is an initial scaffold; actual loader integration should replace the
placeholder document loading once loader APIs are finalized/refactored.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from rag_ed.graphs import CourseGraph

# TODO: import real loaders when available
# from rag_ed.loaders.canvas import load_canvas_documents
# from rag_ed.loaders.piazza import load_piazza_documents


def _placeholder_canvas_docs(canvas_path: str) -> List[Document]:  # pragma: no cover - to be replaced
    return []

def _placeholder_piazza_docs(piazza_path: str) -> List[Document]:  # pragma: no cover - to be replaced
    return []


def build_combined_graph(canvas_path: str, piazza_path: str) -> CourseGraph:
    """Return a combined course graph (placeholder version).

    For now this creates an empty graph; subsequent milestone work will:
    1. Load Canvas documents.
    2. Load Piazza posts.
    3. Insert artifacts into the graph (stable ordering).
    4. Apply semantic edge enrichment (chronological, same_stem, etc.).
    """
    graph = CourseGraph()
    # Placeholder (no-op) until real integration
    return graph
