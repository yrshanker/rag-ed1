"""Utilities for retrieving documents using vector stores."""

from __future__ import annotations

import os
from typing import Literal

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
    """Retrieve documents using a configurable vector store.

    Parameters
    ----------
    canvas_path : str
        Path to the Canvas ``.imscc`` file.
    piazza_path : str
        Path to the Piazza export ``.zip`` file.
    vector_store_type : {"faiss", "in_memory", "chroma"}, optional
        Backend for storing document vectors. Defaults to ``"faiss"``.
    embeddings : langchain_core.embeddings.Embeddings, optional
        Embedding model to use. If omitted, :class:`langchain_openai.embeddings.OpenAIEmbeddings`
        is used.
    persist_directory : str, optional
        Directory for persisting and loading vector indexes.
        Only applies to ``"faiss"`` and ``"chroma"`` stores.
    k : int, optional
        Default number of top documents to retrieve.

    Examples
    --------
    >>> from rag_ed.retrievers.vectorstore import VectorStoreRetriever
    >>> retriever = VectorStoreRetriever(
    ...     "canvas.imscc",
    ...     "piazza.zip",
    ...     vector_store_type="in_memory",
    ... )
    >>> docs = retriever.retrieve("machine learning")
    >>> len(docs)
    5
    """

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
        if not os.path.exists(canvas_path):
            msg = f"Canvas file '{canvas_path}' does not exist."
            raise FileNotFoundError(msg)
        if not os.path.exists(piazza_path):
            msg = f"Piazza file '{piazza_path}' does not exist."
            raise FileNotFoundError(msg)

        canvas_docs = CanvasLoader(canvas_path).load()
        piazza_docs = PiazzaLoader(piazza_path).load()
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
    retriever = VectorStoreRetriever(
        "/Users/work/Downloads/canvas.imscc",
        "/Users/work/Downloads/piazza.zip",
    )
    results = retriever.retrieve("machine learning", k=3)
    for result in results:
        print(result)
