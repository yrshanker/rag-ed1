from pathlib import Path

import langchain_openai.embeddings
from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader
from rag_ed.retrievers.vectorstore import VectorStoreRetriever
from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export


class DummyEmbeddings:
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
            cls, docs: list, embeddings: DummyEmbeddings
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
