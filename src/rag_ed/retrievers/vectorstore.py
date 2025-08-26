from pathlib import Path

import langchain.text_splitter
import langchain.vectorstores
import langchain_core.callbacks.manager
import langchain_core.documents
import langchain_core.retrievers
import langchain_openai.embeddings

from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader


class VectorStoreRetriever(langchain_core.retrievers.BaseRetriever):
    """A retriever that uses CanvasLoader and PiazzaLoader to perform vector retrieval."""

    def __init__(
        self,
        canvas_path: str,
        piazza_path: str,
        in_memory: bool = False,
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

        embeddings = langchain_openai.embeddings.OpenAIEmbeddings()

        if in_memory:
            self.vector_store = (
                langchain.vectorstores.InMemoryVectorStore.from_documents(
                    documents, embeddings
                )
            )
        else:
            self.vector_store = langchain.vectorstores.FAISS.from_documents(
                documents, embeddings
            )
        self.k = k

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


if __name__ == "__main__":
    # Example usage
    retriever = VectorStoreRetriever(
        "/Users/work/Downloads/canvas.imscc", "/Users/work/Downloads/piazza.zip"
    )
    results = retriever.retrieve("machine learning", k=3)
    for result in results:
        print(result)
