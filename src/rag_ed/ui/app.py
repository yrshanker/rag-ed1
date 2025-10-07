"""Gradio UI entry point (bare-bones MVP).

Usage:
    python -m rag_ed.ui.app

Features (MVP):
- Query input
- Canvas & Piazza path inputs
- Retrieval mode (vector|graph|fused)
- Echo toggle (uses EchoLLM when enabled OR when OPENAI_API_KEY missing)
- Edge weights + fusion params (alpha, beta, k, depth)
- Diagnostics JSON for fused mode

Future additions:
- Real combined graph build (currently placeholder)
- Graph visualization
- Score breakdown download / export
- Proper error banners & user guidance
"""
from __future__ import annotations

import os
from typing import Any, Tuple, Dict, List

import gradio as gr
from langchain_core.documents import Document

from rag_ed.retrievers.factory import (
    parse_edge_weights,
    build_for_mode,
)
from rag_ed.graphs.build import build_combined_graph
from rag_ed.retrievers.fusion import CombinedRetriever
from rag_ed.agents.vanilla_rag import EchoLLM  # reuse existing stub


def _echo_answer(query: str, docs: List[Document]) -> str:
    preview = "\n".join(f"[{i+1}] {d.metadata.get('source','')} :: {d.page_content[:160]}" for i, d in enumerate(docs))
    return f"Echo Query: {query}\n\nContext Preview:\n{preview}" if preview else f"Echo Query: {query}\n(No documents)"


def _llm_answer(query: str, docs: List[Document], echo: bool) -> str:
    # For MVP we only implement echo; real LLM integration can wrap RetrievalQA later.
    return _echo_answer(query, docs) if echo else _echo_answer(query, docs)


def run_query(
    query: str,
    canvas_path: str,
    piazza_path: str,
    mode: str,
    echo_mode: bool,
    edge_weight_spec: str,
    alpha: float,
    beta: float,
    k: int,
    depth: int,
    show_diagnostics: bool,
    graph_cache: Dict[str, Any],
    vector_cache: Dict[str, Any],
) -> Tuple[str, List[List[Any]], Any, Dict[str, Any], Dict[str, Any]]:
    """Main callback invoked by Gradio.

    Returns
    -------
    answer_markdown, context_rows, diagnostics_json, graph_cache, vector_cache
    """
    # Basic validation
    if not query.strip():
        return "(Enter a query)", [], [], graph_cache, vector_cache
    # We allow empty paths for now (vector mode may load internal fixtures later); placeholder.

    edge_weights = parse_edge_weights(edge_weight_spec)

    # Build (placeholder) graph for modes needing it. Caching key: tuple of paths.
    graph = None
    cache_key = f"{canvas_path}|{piazza_path}"
    if mode in {"graph", "fused"}:
        graph = graph_cache.get(cache_key)
        if graph is None:
            graph = build_combined_graph(canvas_path, piazza_path)
            graph_cache[cache_key] = graph

    # Build retriever for selected mode.
    retriever, _graph_obj = build_for_mode(
        mode,
        canvas_path=canvas_path,
        piazza_path=piazza_path,
        graph=graph,
        edge_weights=edge_weights or None,
        allowed_kinds=None,
        alpha=alpha,
        beta=beta,
        k=int(k),
        max_graph_depth=int(depth),
    )

    diagnostics = []
    docs: List[Document] = []
    if mode == "fused" and isinstance(retriever, CombinedRetriever) and show_diagnostics:
        diagnostics = retriever.retrieve_with_diagnostics(query)
        # Build doc list in same order as diagnostics
        docs = []
        for item in diagnostics:
            # We don't keep a direct mapping to Document objects here (MVP) so run a plain retrieve
            # to gather docs; ordering may differ if tie-breakers changed; acceptable for MVP.
            pass
        # Use simple retrieve for actual docs to show in context list
        docs = retriever.retrieve(query)
    else:
        # Non-diagnostics path
        if hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)  # type: ignore[attr-defined]
        else:  # BaseRetriever compatibility
            docs = retriever._get_relevant_documents(query, run_manager=None)  # type: ignore[attr-defined]

    answer = _llm_answer(query, docs, echo_mode or (os.environ.get("OPENAI_API_KEY") is None))

    context_rows = [
        [i + 1, d.metadata.get("source", ""), d.page_content[:180]] for i, d in enumerate(docs)
    ]

    return answer, context_rows, diagnostics, graph_cache, vector_cache


def launch_app() -> gr.Blocks:
    with gr.Blocks(title="RAG Explorer", theme="soft") as demo:
        gr.Markdown("# RAG Explorer (MVP)\nExperiment with vector / graph / fused retrieval.")

        with gr.Row():
            with gr.Column(scale=1):
                canvas_path = gr.Textbox(label="Canvas IMSCC Path", placeholder="/path/to/course.imscc")
                piazza_path = gr.Textbox(label="Piazza ZIP Path", placeholder="/path/to/piazza.zip")
                query = gr.Textbox(label="Query", lines=2, placeholder="Ask a question about the course…")
                mode = gr.Dropdown([
                    "vector",
                    "graph",
                    "fused",
                ], value="vector", label="Retrieval Mode")
                echo_mode = gr.Checkbox(value=True, label="Echo Mode (fallback if no API key)")
                edge_weight_spec = gr.Textbox(label="Edge Weights (kind:weight,...)", placeholder="same_stem:5,chronological:1")
                with gr.Accordion("Fusion Parameters", open=False):
                    alpha = gr.Number(value=1.0, label="Alpha (vector)")
                    beta = gr.Number(value=1.0, label="Beta (graph)")
                    k = gr.Number(value=5, label="k (top docs)")
                    depth = gr.Number(value=1, label="Graph Depth")
                    show_diagnostics = gr.Checkbox(value=True, label="Show Diagnostics (fused only)")
                submit = gr.Button("Run Retrieval", variant="primary")
            with gr.Column(scale=2):
                answer = gr.Markdown(label="Answer")
                context_table = gr.Dataframe(
                    headers=["Rank", "Source", "Preview"],
                    datatype=["number", "str", "str"],
                    interactive=False,
                    row_count=(0, "dynamic"),
                    label="Context",
                )
                diagnostics_json = gr.JSON(label="Diagnostics (Fused)")

        graph_cache = gr.State({})
        vector_cache = gr.State({})

        submit.click(
            fn=run_query,
            inputs=[
                query,
                canvas_path,
                piazza_path,
                mode,
                echo_mode,
                edge_weight_spec,
                alpha,
                beta,
                k,
                depth,
                show_diagnostics,
                graph_cache,
                vector_cache,
            ],
            outputs=[
                answer,
                context_table,
                diagnostics_json,
                graph_cache,
                vector_cache,
            ],
        )
    return demo


def main():  # pragma: no cover - manual invocation
    demo = launch_app()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
