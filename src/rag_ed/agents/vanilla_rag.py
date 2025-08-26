"""Simple one-step retrieval agent.

The module provides a thin wrapper around :class:`~rag_ed.retrievers.vectorstore`
to perform a single retrieval and answer a query. File paths are supplied by the
caller; no paths are hard coded within the module.
"""

import argparse

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from rag_ed.retrievers.vectorstore import VectorStoreRetriever


def one_step_retrieval(query: str, *, canvas_path: str, piazza_path: str) -> str:
    """Answer ``query`` using a single retrieval step.

    Parameters
    ----------
    query:
        User question to answer.
    canvas_path:
        Path to a Canvas ``.imscc`` export.
    piazza_path:
        Path to a Piazza ``.zip`` export.

    Returns
    -------
    str
        The answer returned by the language model.
    """

    retriever = VectorStoreRetriever(
        canvas_path=canvas_path, piazza_path=piazza_path, vector_store_type="in_memory"
    )
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7, model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=retriever.vector_store,
    )
    return qa.run(query)


def main() -> None:
    """CLI entry point for one-step retrieval."""
    parser = argparse.ArgumentParser(description="Run one-step retrieval")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--canvas", required=True, help="Path to Canvas .imscc file")
    parser.add_argument("--piazza", required=True, help="Path to Piazza export .zip")
    args = parser.parse_args()
    print(
        one_step_retrieval(args.query, canvas_path=args.canvas, piazza_path=args.piazza)
    )


if __name__ == "__main__":
    main()
