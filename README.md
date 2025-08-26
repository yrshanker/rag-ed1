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

### Self-querying agent

Set the ``CANVAS_PATH`` and ``PIAZZA_PATH`` environment variables to point to
course exports, then query the agent:

```python
import os
from rag_ed.agents import self_querying

os.environ["CANVAS_PATH"] = "canvas.imscc"
os.environ["PIAZZA_PATH"] = "piazza.zip"
print(self_querying.run_agent("overview of week one and then week two"))
```

