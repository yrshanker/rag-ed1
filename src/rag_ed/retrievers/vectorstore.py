"""Utilities for retrieving documents using vector stores."""

from __future__ import annotations

import os
from typing import Any, Literal
from pathlib import Path

import langchain.text_splitter
import langchain.vectorstores
import langchain_core.callbacks.manager
import langchain_core.documents
import langchain_core.embeddings
import langchain_core.retrievers
import langchain_openai.embeddings

from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader


VectorStoreType = Literal["faiss", "in_memory", "chroma"]


class VectorStoreRetriever(langchain_core.retrievers.BaseRetriever):
    """
    Retrieve documents using a configurable vector store backend.

    This retriever supports multiple vector store types, custom embedding models,
    and optional persistence for scalable and repeatable retrieval workflows.

    Parameters
    ----------
    canvas_path : str
        Path to the Canvas `.imscc` file containing course content.
    piazza_path : str
        Path to the Piazza export `.zip` file containing forum posts.
    vector_store_type : {"faiss", "in_memory", "chroma"}, optional
        Type of vector store backend to use:
            - "faiss": Fast, persistent disk-based index (default).
            - "in_memory": Lightweight, non-persistent in-memory index.
            - "chroma": Persistent disk-based index with advanced features.
    embeddings : langchain_core.embeddings.Embeddings, optional
        Embedding model instance for converting text to vectors. If not provided,
        uses `langchain_openai.embeddings.OpenAIEmbeddings` by default.
    persist_directory : str, optional
        Directory path for saving/loading persistent vector indexes. Only applies
        to "faiss" and "chroma" backends. If omitted, indexes are not persisted.
    k : int, optional
        Default number of top documents to retrieve per query. Defaults to 5.

    Examples
    --------
    Basic usage with in-memory store:
    >>> from rag_ed.retrievers.vectorstore import VectorStoreRetriever
    >>> retriever = VectorStoreRetriever(
    ...     "canvas.imscc",
    ...     "piazza.zip",
    ...     vector_store_type="in_memory",
    ... )
    >>> docs = retriever.retrieve("machine learning")
    >>> print(len(docs))
    5

    Using a custom embedding model and persistent FAISS store:
    >>> from langchain_openai.embeddings import OpenAIEmbeddings
    >>> retriever = VectorStoreRetriever(
    ...     "canvas.imscc",
    ...     "piazza.zip",
    ...     vector_store_type="faiss",
    ...     embeddings=OpenAIEmbeddings(),
    ...     persist_directory="./faiss_index",
    ...     k=10,
    ... )
    >>> docs = retriever.retrieve("deep learning")
    >>> print([doc.page_content for doc in docs])
    """

    vector_store: Any
    k: int

    def __init__(
        self,
        canvas_path: str,
        piazza_path: str,
        *,
        vector_store_type: VectorStoreType = "faiss",
        embeddings: langchain_core.embeddings.Embeddings | None = None,
        persist_directory: str | None = None,
        k: int = 5,
    ) -> None:
        """Initialize the retriever with the desired vector storage type.

        Args:
            canvas_path (str): Path to the Canvas .imscc file.
            piazza_path (str): Path to the Piazza zip file.
            in_memory (bool): If True, use in-memory vector storage. Otherwise, use FAISS.
            k (int): Default number of top documents to retrieve.
        """
        canvas = Path(canvas_path)
        if not canvas.is_file():
            msg = f"Canvas file '{canvas_path}' does not exist or is not a file."
            raise FileNotFoundError(msg)
        piazza = Path(piazza_path)
        if not piazza.is_file():
            msg = f"Piazza file '{piazza_path}' does not exist or is not a file."
            raise FileNotFoundError(msg)

        canvas_docs = CanvasLoader(str(canvas)).load()
        piazza_docs = PiazzaLoader(str(piazza)).load()
        documents = canvas_docs + piazza_docs

        text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        documents = text_splitter.split_documents(documents)

        embeddings = embeddings or langchain_openai.embeddings.OpenAIEmbeddings()

        if vector_store_type == "in_memory":
            store = langchain.vectorstores.InMemoryVectorStore.from_documents(
                documents, embeddings
            )
        elif vector_store_type == "faiss":
            if persist_directory and os.path.exists(persist_directory):
                store = langchain.vectorstores.FAISS.load_local(
                    persist_directory,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                store = langchain.vectorstores.FAISS.from_documents(
                    documents, embeddings
                )
                if persist_directory:
                    store.save_local(persist_directory)
        elif vector_store_type == "chroma":
            if persist_directory and os.path.exists(persist_directory):
                store = langchain.vectorstores.Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                )
            else:
                store = langchain.vectorstores.Chroma.from_documents(
                    documents, embeddings, persist_directory=persist_directory
                )
                if persist_directory:
                    store.persist()
        else:  # pragma: no cover - safeguarded by type hints
            msg = f"Unknown vector_store_type: {vector_store_type}"
            raise ValueError(msg)
        object.__setattr__(self, "vector_store", store)
        object.__setattr__(self, "k", k)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: langchain_core.callbacks.manager.CallbackManagerForRetrieverRun,
    ) -> list[langchain_core.documents.Document]:
        return self.vector_store.similarity_search(query, k=self.k)

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[langchain_core.documents.Document]:
        return self.vector_store.similarity_search(query, k=k or self.k)


if __name__ == "__main__":  # pragma: no cover - example usage
    retriever = VectorStoreRetriever("canvas.imscc", "piazza.zip")
    results = retriever.retrieve("machine learning", k=3)
    for result in results:
        print(result)
