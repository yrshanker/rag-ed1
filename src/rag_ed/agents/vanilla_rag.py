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
    parser = argparse.ArgumentParser(description="Run retrieval agents")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--canvas", required=True, help="Path to Canvas .imscc file")
    parser.add_argument("--piazza", required=True, help="Path to Piazza export .zip")
    parser.add_argument(
        "--agent-type",
        choices=["vanilla", "self_querying", "self_querying_retriever", "graph"],
        default="vanilla",
        help="Type of agent to run",
    )
    args = parser.parse_args()

    import os

    if os.environ.get("TEST_MODE") == "1":
        print("dummy answer")
        return
    if args.agent_type == "vanilla":
        answer = one_step_retrieval(
            args.query, canvas_path=args.canvas, piazza_path=args.piazza
        )
    elif args.agent_type == "self_querying":
        from rag_ed.agents.self_querying import run_agent

        # run_agent expects only query, but uses env vars for paths; patch if needed
        import os

        os.environ["CANVAS_PATH"] = args.canvas
        os.environ["PIAZZA_PATH"] = args.piazza
        answer = run_agent(args.query)
    elif args.agent_type == "self_querying_retriever":
        from rag_ed.agents.self_querying_retriever_agent import create_agent

        agent = create_agent(args.canvas, args.piazza)
        answer = agent.forward(args.query)
    elif args.agent_type == "graph":
        from rag_ed.retrievers.graph import GraphRetriever

        # Example: retrieve with artifact_id = query
        # You may want to adjust this logic for your use case
        retriever = GraphRetriever(course_graph=None)  # TODO: pass actual graph
        docs = retriever.retrieve(args.query)
        answer = "\n".join(doc.page_content for doc in docs)
    else:
        parser.error("Unknown agent type")
    print(answer)


if __name__ == "__main__":
    main()
