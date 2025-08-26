# rag-ed

Utilities for building retrieval-augmented applications.

## Quickstart

```python
from rag_ed.retrievers.vectorstore import VectorStoreRetriever

retriever = VectorStoreRetriever(
    "canvas.imscc",
    "piazza.zip",
    vector_store_type="faiss",  # "in_memory" or "chroma"
    persist_directory="./index",
)
docs = retriever.retrieve("machine learning")
```

`VectorStoreRetriever` accepts custom embedding models via dependency
injection and can persist FAISS or Chroma indexes to disk for reuse.

## Config

The Canvas API loader authenticates using a bearer token. Set the
``CANVAS_API_TOKEN`` environment variable before using
``CanvasAPILoader``:

```bash
export CANVAS_API_TOKEN="<your-canvas-token>"
```

