"""Self-querying agent for iterative retrieval.

The agent decomposes a user's question into simpler sub-queries and retrieves
relevant context for each step using :class:`~rag_ed.retrievers.vectorstore.VectorStoreRetriever`.
All configuration is provided via environment variables so no file paths are
hard coded in the source.
"""

from __future__ import annotations

import os
import re
from typing import List

import langchain_core.documents

from rag_ed.retrievers.vectorstore import VectorStoreRetriever

_RETRIEVER: VectorStoreRetriever | None = None


def _split_query(query: str) -> list[str]:
    """Split ``query`` into individual sub-queries.

    Parameters
    ----------
    query:
        The user's full question.

    Returns
    -------
    list[str]
        A list of simplified sub-queries.
    """

    parts = re.split(r"\b(?:and|then)\b|[?\n]", query, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _get_retriever() -> VectorStoreRetriever:
    """Lazily construct the :class:`VectorStoreRetriever`.

    The retriever is cached at module level. ``CANVAS_PATH`` and ``PIAZZA_PATH``
    environment variables must be set to point to the course exports.
    """

    global _RETRIEVER
    if _RETRIEVER is None:
        canvas_path = os.getenv("CANVAS_PATH")
        piazza_path = os.getenv("PIAZZA_PATH")
        if not canvas_path or not piazza_path:
            msg = "Environment variables CANVAS_PATH and PIAZZA_PATH must be set"
            raise RuntimeError(msg)
        _RETRIEVER = VectorStoreRetriever(
            canvas_path, piazza_path, vector_store_type="in_memory"
        )
    return _RETRIEVER


def run_agent(query: str) -> str:
    """Answer ``query`` using iterative retrieval.

    Examples
    --------
    >>> os.environ["CANVAS_PATH"] = "canvas.imscc"
    >>> os.environ["PIAZZA_PATH"] = "piazza.zip"
    >>> run_agent("topic one and then topic two")
    'Sub-query: topic one\n...'
    """

    retriever = _get_retriever()
    responses: List[str] = []
    for sub_query in _split_query(query):
        docs: list[langchain_core.documents.Document] = retriever.retrieve(sub_query, 5)
        docs_text = "\n".join(doc.page_content for doc in docs)
        responses.append(f"Sub-query: {sub_query}\n{docs_text}")
    return "\n\n".join(responses)
