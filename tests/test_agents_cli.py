import subprocess
import sys
import os
import pytest


@pytest.mark.parametrize(
    "agent_type,expected_output",
    [
        ("vanilla", "dummy answer"),
        ("self_querying", "dummy answer"),
        ("self_querying_retriever", "dummy answer"),
        ("graph", "dummy answer"),
    ],
)
def test_cli_agents(monkeypatch, agent_type, expected_output):
    # Patch agent logic to always return 'dummy answer'
    monkeypatch.setattr(
        "rag_ed.agents.vanilla_rag.one_step_retrieval", lambda *a, **kw: "dummy answer"
    )
    monkeypatch.setattr(
        "rag_ed.agents.self_querying.run_agent", lambda *a, **kw: "dummy answer"
    )

    class DummyAgent:
        def forward(self, query):
            return "dummy answer"

    monkeypatch.setattr(
        "rag_ed.agents.self_querying_retriever_agent.create_agent",
        lambda *a, **kw: DummyAgent(),
    )

    class DummyGraphRetriever:
        def retrieve(self, artifact_id):
            class DummyDoc:
                page_content = "dummy answer"

            return [DummyDoc()]

    monkeypatch.setattr(
        "rag_ed.retrievers.graph.GraphRetriever", lambda *a, **kw: DummyGraphRetriever()
    )
    # Run CLI via subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
    import tempfile

    env["TEST_MODE"] = "1"
    with (
        tempfile.NamedTemporaryFile(suffix=".imscc") as canvas_file,
        tempfile.NamedTemporaryFile(suffix=".zip") as piazza_file,
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.rag_ed.agents.vanilla_rag",
                "test",  # positional query argument
                "--agent-type",
                agent_type,
                "--canvas",
                canvas_file.name,
                "--piazza",
                piazza_file.name,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert expected_output in result.stdout
