"""Retriever factory utilities for UI / CLI composition.

This module centralizes construction of vector, graph, and fused retrievers so
other entry points (CLI, Gradio UI) do not duplicate wiring logic.

Design goals:
- Deterministic behavior for tests.
- Lazy building: only build what caller requests.
- Minimal dependencies: avoid importing gradio or UI libs here.
"""
from __future__ import annotations

from typing import Tuple, Optional, Dict, Set

from rag_ed.retrievers.vectorstore import VectorStoreRetriever
from rag_ed.retrievers.graph import GraphRetriever
from rag_ed.retrievers.fusion import CombinedRetriever, FusionConfig
from rag_ed.graphs import CourseGraph


def parse_edge_weights(spec: str | None) -> Dict[str, float]:
    """Parse a comma-separated ``kind:weight`` specification into a dict.

    Invalid tokens are ignored silently; caller can surface a warning if needed.
    """
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        kind, raw_weight = token.split(":", 1)
        try:
            out[kind.strip()] = float(raw_weight)
        except ValueError:
            continue
    return out


def build_vector_retriever(canvas_path: str, piazza_path: str) -> VectorStoreRetriever:
    return VectorStoreRetriever(
        canvas_path=canvas_path,
        piazza_path=piazza_path,
        vector_store_type="in_memory",
    )


def build_graph_retriever(
    graph: CourseGraph,
    *,
    edge_weights: Dict[str, float] | None = None,
    allowed_kinds: Set[str] | None = None,
    max_depth: int = 1,
) -> GraphRetriever:
    return GraphRetriever(
        course_graph=graph,
        max_depth=max_depth,
        allowed_kinds=allowed_kinds,
        edge_weights=edge_weights,
    )


def build_fused_retriever(
    vector: VectorStoreRetriever,
    graph: GraphRetriever,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 5,
    max_graph_depth: int = 1,
) -> CombinedRetriever:
    cfg = FusionConfig(alpha=alpha, beta=beta, k=k, max_graph_depth=max_graph_depth)
    return CombinedRetriever(vector_retriever=vector, graph_retriever=graph, config=cfg)


class RetrieverBuildError(RuntimeError):
    pass


def build_for_mode(
    mode: str,
    *,
    canvas_path: str,
    piazza_path: str,
    graph: CourseGraph | None = None,
    edge_weights: Dict[str, float] | None = None,
    allowed_kinds: Set[str] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 5,
    max_graph_depth: int = 1,
):
    """Return a retriever (and optionally the graph) for a given ``mode``.

    Parameters
    ----------
    mode: {'vector','graph','fused'}
        Retrieval strategy.
    graph:
        Required for 'graph' and 'fused' modes (caller builds it externally until
        a combined graph builder is implemented).
    """
    mode = mode.lower()
    if mode == "vector":
        vect = build_vector_retriever(canvas_path, piazza_path)
        return vect, None
    if mode in {"graph", "fused"}:
        if graph is None:
            raise RetrieverBuildError("Graph instance required for graph/fused modes.")
        graph_ret = build_graph_retriever(
            graph,
            edge_weights=edge_weights,
            allowed_kinds=allowed_kinds,
            max_depth=max_graph_depth,
        )
        if mode == "graph":
            return graph_ret, graph
        vect = build_vector_retriever(canvas_path, piazza_path)
        fused = build_fused_retriever(
            vect,
            graph_ret,
            alpha=alpha,
            beta=beta,
            k=k,
            max_graph_depth=max_graph_depth,
        )
        return fused, graph
    raise RetrieverBuildError(f"Unsupported retrieval mode: {mode}")
