from pathlib import Path
import os
import langchain_core.documents
from langchain_core.embeddings import Embeddings
import pytest
import langchain_openai.embeddings
import rag_ed.retrievers.vectorstore
from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader
from rag_ed.retrievers.vectorstore import VectorStoreRetriever
from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_vector_store_retriever(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        langchain_openai.embeddings, "OpenAIEmbeddings", DummyEmbeddings
    )

    class DummyVectorStore:
        def __init__(self, docs: list) -> None:
            self._docs = docs

        @classmethod
        def from_documents(
            cls, docs: list, embeddings: Embeddings
        ) -> "DummyVectorStore":
            return cls(docs)

        def similarity_search(self, query: str, k: int) -> list:
            return self._docs[:k]

    canvas_path = generate_imscc(
        tmp_path / "canvas_sample.imscc", title="canvas_sample"
    )
    piazza_path = generate_piazza_export(tmp_path / "piazza_sample.zip")
    docs = CanvasLoader(str(canvas_path)).load() + PiazzaLoader(str(piazza_path)).load()
    vector_store = DummyVectorStore.from_documents(docs, DummyEmbeddings())
    retriever = VectorStoreRetriever.__new__(VectorStoreRetriever)
    object.__setattr__(retriever, "vector_store", vector_store)
    object.__setattr__(retriever, "k", 1)
    results = retriever.retrieve("Hello", k=1)
    assert len(results) == 1


def test_custom_embeddings_and_inmemory(monkeypatch, tmp_path: Path) -> None:
    """Ensure custom embeddings and in-memory store are used."""

    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:  # noqa: D401
            return [[1.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:  # noqa: D401
            return [1.0]

    class DummyVectorStore:
        @classmethod
        def from_documents(
            cls, docs: list, embeddings: Embeddings
        ) -> "DummyVectorStore":
            assert embeddings is dummy_embeddings
            return cls()

        def similarity_search(self, query: str, k: int) -> list:  # noqa: D401
            return []

    dummy_embeddings = DummyEmbeddings()
    monkeypatch.setattr(
        rag_ed.retrievers.vectorstore.langchain.vectorstores,
        "InMemoryVectorStore",
        DummyVectorStore,
        raising=False,
    )

    class Loader:
        def __init__(self, _path: str) -> None:  # noqa: D401
            pass

        def load(self) -> list[langchain_core.documents.Document]:  # noqa: D401
            return [langchain_core.documents.Document(page_content="x")]

    canvas = tmp_path / "c.imscc"
    canvas.write_text("x")
    piazza = tmp_path / "p.zip"
    piazza.write_text("x")
    monkeypatch.setattr(rag_ed.retrievers.vectorstore, "CanvasLoader", Loader)
    monkeypatch.setattr(rag_ed.retrievers.vectorstore, "PiazzaLoader", Loader)

    monkeypatch.setattr(
        langchain_openai.embeddings,
        "OpenAIEmbeddings",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not call")),
    )

    VectorStoreRetriever(
        str(canvas),
        str(piazza),
        vector_store_type="in_memory",
        embeddings=dummy_embeddings,
    )


def test_faiss_persistence(monkeypatch, tmp_path: Path) -> None:
    """FAISS indexes are saved and loaded from disk."""

    class DummyFAISS:
        saved: list[str] = []
        loaded: list[str] = []

        @classmethod
        def from_documents(cls, docs: list, embeddings: Embeddings) -> "DummyFAISS":
            return cls()

        def save_local(self, directory: str) -> None:  # noqa: D401
            self.__class__.saved.append(directory)
            os.makedirs(directory, exist_ok=True)

        @classmethod
        def load_local(
            cls,
            directory: str,
            embeddings: Embeddings,
            allow_dangerous_deserialization: bool,
        ) -> "DummyFAISS":
            cls.loaded.append(directory)
            return cls()

        def similarity_search(self, query: str, k: int) -> list:  # noqa: D401
            return []

    monkeypatch.setattr(
        rag_ed.retrievers.vectorstore.langchain.vectorstores, "FAISS", DummyFAISS
    )

    class Loader:
        def __init__(self, _path: str) -> None:  # noqa: D401
            pass

        def load(self) -> list[langchain_core.documents.Document]:  # noqa: D401
            return [langchain_core.documents.Document(page_content="x")]

    canvas = tmp_path / "c.imscc"
    canvas.write_text("x")
    piazza = tmp_path / "p.zip"
    piazza.write_text("x")
    persist_dir = tmp_path / "idx"
    monkeypatch.setattr(rag_ed.retrievers.vectorstore, "CanvasLoader", Loader)
    monkeypatch.setattr(rag_ed.retrievers.vectorstore, "PiazzaLoader", Loader)

    VectorStoreRetriever(
        str(canvas),
        str(piazza),
        vector_store_type="faiss",
        embeddings=DummyEmbeddings(),
        persist_directory=str(persist_dir),
    )
    VectorStoreRetriever(
        str(canvas),
        str(piazza),
        vector_store_type="faiss",
        embeddings=DummyEmbeddings(),
        persist_directory=str(persist_dir),
    )
    assert DummyFAISS.saved == [str(persist_dir)]
    assert DummyFAISS.loaded == [str(persist_dir)]


def test_chroma_persistence(monkeypatch, tmp_path: Path) -> None:
    """Chroma indexes are saved and reloaded."""

    class DummyChroma:
        saved: list[str] = []
        loaded: list[str] = []

        def __init__(
            self,
            persist_directory: str | None = None,
            embedding_function: DummyEmbeddings | None = None,
        ) -> None:  # noqa: D401
            if persist_directory:
                self.__class__.loaded.append(persist_directory)

        @classmethod
        def from_documents(
            cls,
            docs: list,
            embeddings: DummyEmbeddings,
            persist_directory: str | None = None,
        ) -> "DummyChroma":
            instance = cls(persist_directory, embeddings)
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                cls.saved.append(persist_directory)
            return instance

        def persist(self) -> None:  # noqa: D401
            pass

        def similarity_search(self, query: str, k: int) -> list:  # noqa: D401
            return []

    monkeypatch.setattr(
        rag_ed.retrievers.vectorstore.langchain.vectorstores, "Chroma", DummyChroma
    )

    class Loader:
        def __init__(self, _path: str) -> None:  # noqa: D401
            pass

        def load(self) -> list[langchain_core.documents.Document]:  # noqa: D401
            return [langchain_core.documents.Document(page_content="x")]

    canvas = tmp_path / "c.imscc"
    canvas.write_text("x")
    piazza = tmp_path / "p.zip"
    piazza.write_text("x")
    persist_dir = tmp_path / "chroma"
    monkeypatch.setattr(rag_ed.retrievers.vectorstore, "CanvasLoader", Loader)
    monkeypatch.setattr(rag_ed.retrievers.vectorstore, "PiazzaLoader", Loader)

    VectorStoreRetriever(
        str(canvas),
        str(piazza),
        vector_store_type="chroma",
        embeddings=DummyEmbeddings(),
        persist_directory=str(persist_dir),
    )
    VectorStoreRetriever(
        str(canvas),
        str(piazza),
        vector_store_type="chroma",
        embeddings=DummyEmbeddings(),
        persist_directory=str(persist_dir),
    )
    assert DummyChroma.saved == [str(persist_dir)]
    assert DummyChroma.loaded[-1] == str(persist_dir)


def test_vector_store_retriever_missing_files() -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Canvas file 'missing.imscc' does not exist.",
    ):
        VectorStoreRetriever("missing.imscc", "missing.zip")
