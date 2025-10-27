"""Piazza API loader."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from piazza_api import Piazza


class PiazzaAPILoader(BaseLoader):
    """Load posts from Piazza via the unofficial API."""

    def __init__(
        self,
        network_id: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize the loader.

        Args:
            network_id: Piazza network identifier.
            email: Login email. Falls back to ``PIAZZA_API_EMAIL`` environment variable.
            password: Login password. Falls back to ``PIAZZA_API_PASSWORD`` environment variable.
        """
        self.network_id = network_id
        self.email = email or os.environ["PIAZZA_API_EMAIL"]
        self.password = password or os.environ["PIAZZA_API_PASSWORD"]

    def load(self) -> List[Document]:  # type: ignore[override]
        """Fetch all posts visible to the authenticated user."""
        piazza = Piazza()
        piazza.user_login(email=self.email, password=self.password)
        network = piazza.network(self.network_id)
        posts: Iterable[Dict[str, Any]] = network.iter_all_posts()
        return [self._post_to_doc(post) for post in posts]

    def _post_to_doc(self, post: Dict[str, Any]) -> Document:
        """Convert a Piazza post dictionary to a Document."""
        history = post.get("history", [])
        latest = history[0] if history else {}
        subject = latest.get("subject", "")
        content = latest.get("content", "")
        text = f"{subject}\n\n{content}".strip()
        metadata: Dict[str, Any] = {"network_id": self.network_id}
        if "id" in post:
            metadata["id"] = post["id"]
        if "nr" in post:
            metadata["nr"] = post["nr"]
            metadata["source"] = (
                f"https://piazza.com/class/{self.network_id}/post/{post['nr']}"
            )
        timestamp = post.get("created") or latest.get("created")
        if timestamp:
            metadata["timestamp"] = timestamp
        return Document(page_content=text, metadata=metadata)
