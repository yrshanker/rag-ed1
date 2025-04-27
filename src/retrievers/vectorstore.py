from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, InMemoryVectorStore

from ..loaders.canvas import CanvasLoader
from ..loaders.piazza import PiazzaLoader


class VectorStoreRetriever:
    """
    A retriever that uses CanvasLoader and PiazzaLoader to load documents and perform vector retrieval.
    """

    def __init__(
        self,
        canvas_path: str,
        piazza_path: str,
        in_memory: bool = False,
    ):
        """
        Initialize the retriever with the desired vector storage type.

        Args:
            canvas_path (str): Path to the Canvas .imscc file.
            piazza_path (str): Path to the Piazza zip file.
            in_memory (bool): If True, use in-memory vector storage. Otherwise, use FAISS.
            hf_embedding_model (str): Hugging Face model name for embeddings.
        """

        # Load documents from Canvas and Piazza
        canvas_docs = CanvasLoader(canvas_path).load()
        piazza_docs = PiazzaLoader(piazza_path).load()
        documents = canvas_docs + piazza_docs

        # Chunk the documents using langchain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        documents = text_splitter.split_documents(documents)

        # Create embeddings object that uses local api endpoint
        embeddings = OpenAIEmbeddings()

        # Create the database
        if in_memory:
            self.vector_store = InMemoryVectorStore.from_documents(
                documents, embeddings
            )
        else:
            self.vector_store = FAISS.from_documents(documents, embeddings)

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve the top-k documents matching the query.

        Args:
            query (str): The query string.
            k (int): The number of top documents to retrieve.

        Returns:
            list: A list of retrieved documents.
        """

        return self.vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    # Example usage
    retriever = VectorStoreRetriever(
        "/Users/work/Downloads/canvas.imscc", "/Users/work/Downloads/piazza.zip"
    )
    results = retriever.retrieve("machine learning", k=3)
    for result in results:
        print(result)
