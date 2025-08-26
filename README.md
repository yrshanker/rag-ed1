# rag-ed

Utilities for building retrieval-augmented applications.

## Development

Install the project with development dependencies and run the quality checks:

```bash
pip install -e .[dev]
ruff .
black --check .
mypy .
pytest
pip-audit
```
