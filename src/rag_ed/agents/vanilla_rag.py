"""Simple one-step retrieval agent.

The module provides a thin wrapper around :class:`~rag_ed.retrievers.vectorstore`
to perform a single retrieval and answer a query. File paths are supplied by the
caller; no paths are hard coded within the module.

Echo / pass-through mode
------------------------
For fast, dependency-free testing you can enable *echo mode* via the
``--echo`` CLI flag or the ``ECHO_MODE=1`` environment variable. In echo mode
the retrieval stack is still constructed and invoked, but the language model
is replaced with a lightweight in-process stub that simply returns the input
prompt (no network calls, no cost). This differs from ``TEST_MODE=1`` which
short-circuits the entire pipeline and prints ``dummy answer`` without
performing retrieval.
"""

import argparse

import langchain_core.embeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI  # Updated per deprecation notice
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional

from rag_ed.embeddings import PassThroughEmbeddings
from rag_ed.retrievers.vectorstore import VectorStoreRetriever


class EchoLLM(LLM):
    """A minimal LLM implementation that echoes the final prompt.

    Useful for deterministic, offline testing of the retrieval + prompt plumbing.
    """

    @property
    def _llm_type(self) -> str:  # pragma: no cover - trivial
        return "echo"

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None
    ) -> str:  # type: ignore[override]
        # Ignore stop tokens for simplicity; echo raw prompt.
        return prompt

    @property
    def _identifying_params(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"mode": "echo"}


def one_step_retrieval(
    query: str,
    *,
    canvas_path: str,
    piazza_path: str,
    pass_through: bool = False,
    echo: bool = False,
    embeddings: langchain_core.embeddings.Embeddings | None = None,
) -> str:
    """Answer ``query`` using a single retrieval step.

    Parameters
    ----------
    query:
        User question to answer.
    canvas_path:
        Path to a Canvas ``.imscc`` export.
    piazza_path:
        Path to a Piazza ``.zip`` export.
    pass_through:
        If ``True``, return retrieved documents directly instead of calling an
        LLM.
    embeddings:
        Optional embedding model. Defaults to
        :class:`langchain_openai.embeddings.OpenAIEmbeddings`.

    Returns
    -------
    str
        The answer returned by the language model or concatenated documents
        when ``pass_through`` is ``True``.
    """

    # Default to pass-through embeddings for offline modes to avoid network calls.
    if embeddings is None and (pass_through or echo):
        embeddings = PassThroughEmbeddings()

    retriever = VectorStoreRetriever(
        canvas_path=canvas_path,
        piazza_path=piazza_path,
        vector_store_type="in_memory",
        embeddings=embeddings,
    )
    if pass_through:
        docs = retriever.retrieve(query)
        return "\n".join(doc.page_content for doc in docs)
    llm = EchoLLM() if echo else OpenAI(temperature=0.7, model_name="gpt-4o-mini")
    # Pass the retriever itself (a BaseRetriever) rather than the underlying
    # raw vector store so LangChain's type validation succeeds. The previous
    # implementation passed the internal vector store which is not a
    # BaseRetriever, causing ValidationError in tests that monkeypatch a
    # lightweight dummy implementation.
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    # Use modern invoke API instead of deprecated .run()
    # Prefer the modern invoke API, but support older objects that implement
    # run() for backwards compatibility in tests and third-party mocks.
    if hasattr(qa, "invoke"):
        result = qa.invoke({"query": query})
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result  # type: ignore[return-value]
    # Fallback to legacy run()
    if hasattr(qa, "run"):
        return qa.run(query)
    raise RuntimeError("RetrievalQA object has neither 'invoke' nor 'run' methods")


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
    parser.add_argument(
        "--pass-through",
        action="store_true",
        help="Return retrieved documents without calling the LLM.",
    )
    parser.add_argument(
        "--graph-allowed-kinds",
        help=(
            "Comma-separated list of graph edge kinds to traverse when using --agent-type graph. "
            "If omitted, all edge kinds are considered."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "graph", "fused"],
        default="vector",
        help="Retrieval backend to use inside vanilla agent (vector = default vector store, graph = graph traversal only, fused = vector+graph fusion).",
    )
    parser.add_argument(
        "--graph-edge-weights",
        help="Edge kind weights for graph/fused retrieval as kind:weight[,kind:weight]. Example: same_stem:5,chronological:1",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=1.0,
        help="Alpha coefficient for vector rank component in fused retrieval scoring.",
    )
    parser.add_argument(
        "--fusion-beta",
        type=float,
        default=1.0,
        help="Beta coefficient for graph weight component in fused retrieval scoring.",
    )
    parser.add_argument(
        "--fusion-k",
        type=int,
        default=5,
        help="Number of documents to return from fused retrieval stage before LLM consumption.",
    )
    parser.add_argument(
        "--fusion-max-graph-depth",
        type=int,
        default=1,
        help="Maximum graph traversal depth for fused/graph retrieval modes.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable echo (pass-through) mode: retrieval runs but LLM output is the input prompt.",
    )
    args = parser.parse_args()

    import os

    if os.environ.get("TEST_MODE") == "1":
        print("dummy answer")
        return
    # Determine echo mode precedence: CLI flag > env var
    echo_mode = bool(args.echo or os.environ.get("ECHO_MODE") == "1")
    if args.agent_type == "vanilla":
        # Build retrieval backend based on --retrieval-mode
        if args.retrieval_mode == "vector":
            embeddings = PassThroughEmbeddings() if (args.pass_through or echo_mode) else None
            answer = one_step_retrieval(
                args.query,
                canvas_path=args.canvas,
                piazza_path=args.piazza,
                pass_through=args.pass_through,
                echo=echo_mode,
                embeddings=embeddings,
            )
        elif args.retrieval_mode in {"graph", "fused"}:
            from rag_ed.graphs import CourseGraph
            from rag_ed.retrievers.graph import GraphRetriever
            from rag_ed.retrievers.vectorstore import VectorStoreRetriever
            from rag_ed.retrievers.fusion import CombinedRetriever, FusionConfig
            from langchain.chains import RetrievalQA

            # NOTE: Placeholder graph construction until unified build util exists.
            course_graph = CourseGraph()  # TODO: build from canvas/piazza like vector store.
            allowed = None
            if args.graph_allowed_kinds:
                allowed = {k.strip() for k in args.graph_allowed_kinds.split(",") if k.strip()}
            edge_weights = {}
            if args.graph_edge_weights:
                for part in args.graph_edge_weights.split(","):
                    if not part.strip():
                        continue
                    if ":" not in part:
                        continue
                    k, v = part.split(":", 1)
                    try:
                        edge_weights[k.strip()] = float(v)
                    except ValueError:
                        pass  # silently ignore malformed entries
            graph_retriever = GraphRetriever(
                course_graph,
                max_depth=args.fusion_max_graph_depth,
                allowed_kinds=allowed,
                edge_weights=edge_weights or None,
            )
            if args.retrieval_mode == "graph":
                # Direct graph traversal answer (concatenate docs)
                docs = graph_retriever.retrieve(args.query, max_depth=args.fusion_max_graph_depth)
                answer = "\n".join(doc.page_content for doc in docs)
            else:  # fused
                vector_retriever = VectorStoreRetriever(
                    canvas_path=args.canvas,
                    piazza_path=args.piazza,
                    vector_store_type="in_memory",
                )
                fusion_config = FusionConfig(
                    alpha=args.fusion_alpha,
                    beta=args.fusion_beta,
                    max_graph_depth=args.fusion_max_graph_depth,
                    k=args.fusion_k,
                )
                fused_retriever = CombinedRetriever(
                    vector_retriever=vector_retriever, graph_retriever=graph_retriever, config=fusion_config
                )
                llm = EchoLLM() if echo_mode else OpenAI(temperature=0.7, model_name="gpt-4o-mini")
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=fused_retriever,
                )
                if hasattr(qa, "invoke"):
                    result = qa.invoke({"query": args.query})
                    if isinstance(result, dict) and "result" in result:
                        answer = result["result"]
                    else:
                        answer = result  # type: ignore[assignment]
                else:
                    answer = qa.run(args.query)
        else:
            raise ValueError(f"Unknown retrieval mode: {args.retrieval_mode}")
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
        allowed = None
        if args.graph_allowed_kinds:
            allowed = {k.strip() for k in args.graph_allowed_kinds.split(",") if k.strip()}
        retriever = GraphRetriever(course_graph=None, allowed_kinds=allowed)  # TODO: pass actual graph
        docs = retriever.retrieve(args.query)
        answer = "\n".join(doc.page_content for doc in docs)
    else:
        parser.error("Unknown agent type")
    print(answer)


if __name__ == "__main__":
    main()
