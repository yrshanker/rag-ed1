"""
This module provides a unified interface for various data loaders used for education
"""

# Re-export the loaders for easier access
from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.piazza import PiazzaLoader

# Re-export the retriever for easier access
from rag_ed.retrievers.vectorstore import VectorStoreRetriever
