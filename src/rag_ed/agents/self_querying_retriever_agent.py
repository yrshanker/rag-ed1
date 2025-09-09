from smolagents import CodeAgent, OpenAIServerModel, Tool

from rag_ed.retrievers.vectorstore import VectorStoreRetriever


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, canvas_path: str, piazza_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.retriever = VectorStoreRetriever(
            canvas_path, piazza_path, vector_store_type="in_memory"
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.retrieve(query, 5)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


def create_retriever_tool(
    canvas_path: str, piazza_path: str, **kwargs
) -> RetrieverTool:
    """Instantiate a :class:`RetrieverTool` with the provided data sources."""

    return RetrieverTool(canvas_path, piazza_path, **kwargs)


def create_agent(canvas_path: str, piazza_path: str, **kwargs) -> CodeAgent:
    """Build a :class:`CodeAgent` that uses the :class:`RetrieverTool`."""

    retriever_tool = create_retriever_tool(canvas_path, piazza_path, **kwargs)
    return CodeAgent(
        tools=[retriever_tool],
        model=OpenAIServerModel(model_id="dummy-model-id"),
        max_steps=4,
        verbosity_level=2,
    )
