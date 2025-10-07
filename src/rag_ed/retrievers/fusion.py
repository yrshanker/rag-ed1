"""Fusion retriever combining vector and graph signals.

Implements a simple composite scoring function:
    score = alpha * (1 / (1 + vector_rank)) + beta * graph_weight

Where:
- vector_rank starts at 0 for the most similar vector result
- graph_weight is the edge weight assigned during graph traversal
- alpha and beta default to 1.0

Deterministic ordering is guaranteed via tie-breaking on document source path then id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List

import langchain_core.documents
import langchain_core.retrievers

from .graph import GraphRetriever


@dataclass
class FusionConfig:
    alpha: float = 1.0
    beta: float = 1.0
    max_graph_depth: int = 1
    k: int = 5  # final number of docs to return


class CombinedRetriever(langchain_core.retrievers.BaseRetriever):
    def __init__(
        self,
        *,
        vector_retriever: langchain_core.retrievers.BaseRetriever,
        graph_retriever: GraphRetriever,
        config: FusionConfig | None = None,
    ) -> None:
        self._vector = vector_retriever
        self._graph = graph_retriever
        self._config = config or FusionConfig()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: langchain_core.callbacks.manager.CallbackManagerForRetrieverRun,  # type: ignore[name-defined]
    ) -> List[langchain_core.documents.Document]:
        return self.retrieve(query)

    def retrieve(self, query: str) -> List[langchain_core.documents.Document]:
        # Vector pass: prefer explicit 'retrieve' to avoid BaseRetriever.invoke tag plumbing.
        if hasattr(self._vector, "retrieve"):
            vector_docs: Sequence[langchain_core.documents.Document] = self._vector.retrieve(query)  # type: ignore[attr-defined]
        else:
            # Fall back to protected API used by BaseRetriever to remain compatible with dummy/simple retrievers in tests.
            vector_docs = self._vector._get_relevant_documents(query, run_manager=None)  # type: ignore[attr-defined]

        # Graph pass uses the query as artifact id.
        graph_docs_with_scores = self._graph.retrieve_with_scores(
            query, max_depth=self._config.max_graph_depth
        )

        # Build score list using integer indices instead of hashing Document objects.
        entries: list[tuple[langchain_core.documents.Document, float, bool]] = []  # (doc, score, is_vector)

        # Vector contribution (rank-based).
        for rank, doc in enumerate(vector_docs):
            vrank_component = self._config.alpha * (1.0 / (1 + rank))
            entries.append((doc, vrank_component, True))

        # Graph contribution: merge with existing vector docs if object identity matches.
        # We'll aggregate scores by id(doc) to avoid hashing Document.
        aggregated: dict[int, tuple[langchain_core.documents.Document, float, bool]] = {}
        for doc, score, is_vec in entries:
            aggregated[id(doc)] = (doc, score, is_vec)

        for gdoc, weight in graph_docs_with_scores:
            comp = self._config.beta * weight
            doc_id = id(gdoc)
            if doc_id in aggregated:
                doc, existing_score, is_vec = aggregated[doc_id]
                aggregated[doc_id] = (doc, existing_score + comp, is_vec)
            else:
                aggregated[doc_id] = (gdoc, comp, False)

        merged = list(aggregated.values())

        def _stable_key(item: tuple[langchain_core.documents.Document, float, bool]):
            doc, score, is_vec = item
            source = doc.metadata.get("source", "")
            vec_priority = 0 if is_vec else 1
            return (-score, vec_priority, source, id(doc))

        ranked = sorted(merged, key=_stable_key)
        return [doc for doc, _, _ in ranked[: self._config.k]]

    # NOTE: Diagnostics method returns structured breakdown for UI/debugging.
    def retrieve_with_diagnostics(self, query: str) -> list[dict]:  # pragma: no cover - UI utility
        if hasattr(self._vector, "retrieve"):
            vector_docs: Sequence[langchain_core.documents.Document] = self._vector.retrieve(query)  # type: ignore[attr-defined]
        else:
            vector_docs = self._vector._get_relevant_documents(query, run_manager=None)  # type: ignore[attr-defined]
        graph_docs_with_scores = self._graph.retrieve_with_scores(
            query, max_depth=self._config.max_graph_depth
        )
        # Build partial maps
        vector_part: dict[int, float] = {}
        for rank, doc in enumerate(vector_docs):
            vector_part[id(doc)] = self._config.alpha * (1.0 / (1 + rank))
        graph_part: dict[int, float] = {id(d): self._config.beta * w for d, w in graph_docs_with_scores}
        # Aggregate
        doc_objs: dict[int, langchain_core.documents.Document] = {id(d): d for d in vector_docs}
        for d, _w in graph_docs_with_scores:
            doc_objs.setdefault(id(d), d)
        diagnostics: list[tuple[langchain_core.documents.Document, float, float, float]] = []
        for doc_id, doc in doc_objs.items():
            v = vector_part.get(doc_id, 0.0)
            g = graph_part.get(doc_id, 0.0)
            total = v + g
            diagnostics.append((doc, v, g, total))
        diagnostics.sort(key=lambda t: (-t[3], 0 if t[1] > 0 else 1, doc.metadata.get("source", ""), id(t[0])))  # type: ignore[name-defined]
        # Truncate to k
        diagnostics = diagnostics[: self._config.k]
        out: list[dict] = []
        for doc, v, g, total in diagnostics:
            out.append(
                {
                    "source": doc.metadata.get("source", ""),
                    "content_preview": doc.page_content[:200],
                    "vector_component": v,
                    "graph_component": g,
                    "score": total,
                }
            )
        return out
