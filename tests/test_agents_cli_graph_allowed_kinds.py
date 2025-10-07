import sys
import tempfile


def test_cli_graph_allowed_kinds_flag(monkeypatch, capsys):
    """Ensure --graph-allowed-kinds is parsed and passed to GraphRetriever."""
    from langchain_core.documents import Document
    from rag_ed.retrievers.graph import GraphRetriever as RealGraphRetriever

    captured_allowed = {}

    class DummyGraph:
        graph = type('G', (), {'nodes': {}, 'neighbors': lambda self, n: []})()

    class DummyGraphRetriever(RealGraphRetriever):  # subclass to keep interface
        def __init__(self, course_graph, *, max_depth: int = 1, allowed_kinds=None):
            super().__init__(course_graph or DummyGraph(), max_depth=max_depth, allowed_kinds=allowed_kinds)
            captured_allowed['value'] = allowed_kinds
        def retrieve(self, artifact_id: str, *, max_depth: int | None = None):  # pragma: no cover
            return [Document(page_content="OK")]  # one dummy doc

    # Patch only the retrievers.graph module. The CLI imports GraphRetriever lazily
    # inside the graph branch, so this suffices.
    monkeypatch.setattr("rag_ed.retrievers.graph.GraphRetriever", DummyGraphRetriever)

    with (
        tempfile.NamedTemporaryFile(suffix=".imscc") as canvas_file,
        tempfile.NamedTemporaryFile(suffix=".zip") as piazza_file,
    ):
        sys.argv = [
            "vanilla_rag.py",
            "node_0",  # artifact id for retrieval
            "--agent-type",
            "graph",
            "--canvas",
            canvas_file.name,
            "--piazza",
            piazza_file.name,
            "--graph-allowed-kinds",
            "same_stem,thread_reply",
        ]
        from rag_ed.agents import vanilla_rag as cli
        cli.main()
        assert captured_allowed['value'] == {"same_stem", "thread_reply"}
        out = capsys.readouterr().out
        assert "OK" in out
