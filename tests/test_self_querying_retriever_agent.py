from rag_ed.agents.self_querying_retriever_agent import create_agent
from smolagents import Tool


class DummyRetrieverTool(Tool):
    name = "retriever"
    description = "Dummy retriever tool for testing."
    inputs = {"query": {"type": "string", "description": "The query to perform."}}
    output_type = "string"

    def forward(self, query):
        return "dummy answer"


def test_create_agent_builds_functioning_agent(monkeypatch):
    monkeypatch.setattr(
        "rag_ed.agents.self_querying_retriever_agent.create_retriever_tool",
        lambda *a, **kw: DummyRetrieverTool(),
    )

    class DummyModel:
        pass

    monkeypatch.setattr(
        "rag_ed.agents.self_querying_retriever_agent.OpenAIServerModel",
        lambda *a, **kw: DummyModel(),
    )
    agent = create_agent("canvas.imscc", "piazza.zip")
    assert hasattr(agent, "tools")
    assert hasattr(agent, "forward") or hasattr(agent, "run")
    # Simulate agent usage
    # Support both list and dict for agent.tools
    tools = agent.tools
    if isinstance(tools, dict):
        tool = next(iter(tools.values()))
    elif isinstance(tools, (list, tuple)):
        tool = tools[0]
    else:
        tool = tools
    result = tool.forward("test query")
    assert result == "dummy answer"
