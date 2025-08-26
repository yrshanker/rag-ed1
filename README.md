# rag-ed

[![CI](https://github.com/yourusername/rag-ed/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/rag-ed/actions/workflows/ci.yml)

## What & Why

**rag-ed** offers utilities and examples for building retrieval-augmented applications. It focuses on simple, composable pieces so you can experiment with RAG pipelines that combine course content from Canvas and Piazza archives.

## Quickstart

```bash
pip install rag-ed
vanilla-rag --canvas path/to/export.imscc --piazza path/to/export.zip "What is due next week?"
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
```

## Config

- **Canvas export**: `.imscc` archive of your course.
- **Piazza export**: `.zip` archive downloaded from Piazza.
- **OPENAI_API_KEY**: authentication token for OpenAI's API.

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
ruff check .
black .
mypy .
pytest
```

## Troubleshooting

- Verify the OpenAI API key is set and valid.
- Ensure the Canvas and Piazza archives are unmodified downloads.
- Run `pip install .[dev]` to obtain all required dependencies.

## License

rag-ed is released under the [MIT License](LICENSE).

