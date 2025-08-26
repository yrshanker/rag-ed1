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

## Graph-based retrieval (experimental)

The `rag_ed.graphs` package uses `networkx` to model relationships among course
artifacts. `GraphRetriever` walks these graphs to fetch documents related to a
given artifact. When constructing graphs from Canvas and Piazza exports, nodes
are grouped by their source directory and linked in chronological order to
retain basic structure.

```python
from langchain_core.documents import Document
from rag_ed.graphs import (
    CourseGraph,
    graph_from_canvas,
    graph_from_piazza,
)
from rag_ed.retrievers.graph import GraphRetriever

graph = CourseGraph()
graph.add_artifact("a", Document(page_content="A"))
graph.add_artifact("b", Document(page_content="B"))
graph.add_relationship("a", "b")

canvas_graph = graph_from_canvas("course.imscc")
piazza_graph = graph_from_piazza("piazza.zip")

retriever = GraphRetriever(graph, max_depth=1)
docs = retriever.retrieve("a")
```

> **Warning**
> Graph-based retrieval is experimental and its APIs may change.

