import argparse

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from rag_ed.retrievers.vectorstore import VectorStoreRetriever


def one_step_retrieval(query: str) -> str:
    """
    Perform one-step retrieval using the specified query.
    Args:
        query (str): The query string.
    Returns:
        str: The answer to the query.
    """
    # Initialize the retriever with sample file paths
    retriever = VectorStoreRetriever(
        canvas_path="/Users/work/Downloads/canvas.imscc",
        piazza_path="/Users/work/Downloads/piazza.zip",
        in_memory=True,
    )
    # Create a RetrievalQA chain using the retriever's vector store
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7, model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=retriever.vector_store,
    )
    answer = qa.run(query)
    return answer


def main() -> None:
    """CLI entry point for one-step retrieval."""
    parser = argparse.ArgumentParser(description="Run one-step retrieval")
    parser.add_argument("query", help="Query string")
    args = parser.parse_args()
    print(one_step_retrieval(args.query))


if __name__ == "__main__":
    main()
