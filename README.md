# rag-ed

[![CI](https://github.com/cmccomb/rag-ed/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/rag-ed/actions/workflows/ci.yml)

## What & Why

**rag-ed** offers utilities and examples for building retrieval-augmented applications. It focuses on simple, composable pieces so you can experiment with RAG pipelines that combine course content from Canvas and Piazza archives.

## Quickstart

```bash
pip install rag-ed
vanilla-rag --canvas path/to/export.imscc --piazza path/to/export.zip "What is due next week?"
# run offline and return retrieved documents
vanilla-rag --pass-through --canvas path/to/export.imscc --piazza path/to/export.zip "Piazza"
```

The CLI expects an OpenAI API key in the `OPENAI_API_KEY` environment variable.

## Examples

```python
from rag_ed.agents.vanilla_rag import one_step_retrieval

answer = one_step_retrieval(
    "What topics are covered in week 2?",
    canvas_path="course.imscc",
    piazza_path="piazza.zip",
)
print(answer)

# offline pass-through mode
from rag_ed.embeddings import PassThroughEmbeddings

docs = one_step_retrieval(
    "Piazza",
    canvas_path="course.imscc",
    piazza_path="piazza.zip",
    pass_through=True,
    embeddings=PassThroughEmbeddings(),
)
print(docs)
```

```python
from rag_ed.loaders.piazza_api import PiazzaAPILoader

posts = PiazzaAPILoader("network_id", email="user@example.com", password="pw").load()
```

```python
from langchain_core.documents import Document
from rag_ed.graphs import CourseGraph
from rag_ed.retrievers.graph import GraphRetriever

graph = CourseGraph()
graph.add_artifact("a", Document(page_content="A"))
graph.add_artifact("b", Document(page_content="B"))
graph.add_relationship("a", "b")
retriever = GraphRetriever(graph)
retriever.retrieve("a")  # returns [Document(page_content="B")]
```

## Config

- **Canvas export**: `.imscc` archive of your course.
- **Piazza export**: `.zip` archive downloaded from Piazza.
- **OPENAI_API_KEY**: authentication token for OpenAI's API.
- **PIAZZA_API_EMAIL / PIAZZA_API_PASSWORD**: credentials for the Piazza API loader.

## CLI / API Reference

### `vanilla-rag`

```
usage: vanilla-rag [-h] --canvas CANVAS --piazza PIAZZA query
```

Runs a single-step retrieval using the provided Canvas and Piazza data.

### Python API

`one_step_retrieval(query, *, canvas_path, piazza_path) -> str`

## Development

```bash
git clone https://github.com/yourusername/rag-ed.git
cd rag-ed
pip install .[dev]
pre-commit install
pre-commit run --files <paths>
pytest
```

## Troubleshooting

- Verify the OpenAI API key is set and valid.
- Ensure the Canvas and Piazza archives are unmodified downloads.
- Run `pip install .[dev]` to obtain all required dependencies.

## License

rag-ed is released under the [MIT License](LICENSE).

