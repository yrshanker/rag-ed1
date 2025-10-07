import sys
import tempfile


def test_cli_vanilla_echo_mode(monkeypatch, capsys):
    """Ensure --echo causes the vanilla agent to return the raw query.

    Runs the CLI entry point in-process so monkeypatching the retriever prevents
    real file/embedding operations. This validates that echo mode routes through
    the RetrievalQA chain with an EchoLLM instead of making network calls.
    """

    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import (
        CallbackManagerForRetrieverRun as _RunMgr,
    )
    from langchain_core.documents import Document

    class DummyRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: _RunMgr
        ):  # pragma: no cover - simple
            return [Document(page_content=f"DOC:{query}")]

        # Provide public retrieve helper mirroring real retriever for completeness.
        def retrieve(self, query: str, k: int | None = None):  # pragma: no cover
            return self._get_relevant_documents(query, run_manager=None)  # type: ignore[arg-type]

    # Patch both original module class and already-imported symbol.
    monkeypatch.setattr(
        "rag_ed.retrievers.vectorstore.VectorStoreRetriever", DummyRetriever
    )
    monkeypatch.setattr(
        "rag_ed.agents.vanilla_rag.VectorStoreRetriever", DummyRetriever
    )

    query = "What is retrieval augmentation?"

    # Create placeholder files (existence only; not parsed due to monkeypatch).
    with (
        tempfile.NamedTemporaryFile(suffix=".imscc") as canvas_file,
        tempfile.NamedTemporaryFile(suffix=".zip") as piazza_file,
    ):
        sys.argv = [
            "vanilla_rag.py",
            query,
            "--agent-type",
            "vanilla",
            "--canvas",
            canvas_file.name,
            "--piazza",
            piazza_file.name,
            "--echo",
        ]
        from rag_ed.agents import vanilla_rag as cli

        cli.main()
        captured = capsys.readouterr()
        # EchoLLM returns the final prompt; ensure original query appears.
        assert query in captured.out