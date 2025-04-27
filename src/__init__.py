"""
This module provides a unified interface for various data loaders used for education
"""

# Re-export the loaders for easier access
from .loaders.canvas import CanvasLoader
from .loaders.piazza import PiazzaLoader

# Re-export the retriever for easier access
from .retrievers.vectorstore import VectorStoreRetriever
