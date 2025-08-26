"""Canvas API loader."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CanvasAPILoader(BaseLoader):
    """Load Canvas course data via the Canvas REST API.

    The loader retrieves assignments, quizzes, and announcements for a course
    using a bearer token for authentication.
    """

    def __init__(
        self, base_url: str, course_id: int, token: Optional[str] = None
    ) -> None:
        """Initialize the loader.

        Args:
            base_url: Base URL of the Canvas instance, e.g. ``"https://canvas.instructure.com"``.
            course_id: The Canvas course identifier.
            token: API token. Falls back to ``CANVAS_API_TOKEN`` environment variable.
        """
        self.base_url = base_url.rstrip("/")
        self.course_id = course_id
        self.token = token or os.environ["CANVAS_API_TOKEN"]

    def load(self) -> List[Document]:  # type: ignore[override]
        """Retrieve assignments, quizzes, and announcements."""
        documents: List[Document] = []
        documents.extend(
            self._load_endpoint(
                f"/api/v1/courses/{self.course_id}/assignments",
                self._assignment_to_doc,
            )
        )
        documents.extend(
            self._load_endpoint(
                f"/api/v1/courses/{self.course_id}/quizzes",
                self._quiz_to_doc,
            )
        )
        documents.extend(
            self._load_endpoint(
                "/api/v1/announcements",
                self._announcement_to_doc,
                params={"context_codes[]": f"course_{self.course_id}"},
            )
        )
        return documents

    def _load_endpoint(
        self,
        endpoint: str,
        converter: Callable[[Dict[str, Any]], Document],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        url = f"{self.base_url}{endpoint}"
        documents: List[Document] = []
        while url:
            response = self._get(url, params=params)
            data = response.json()
            for item in data:
                documents.append(converter(item))
            url = response.links.get("next", {}).get("url")
            params = None
        return documents

    def _get(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.token}"}
        while True:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                self._respect_rate_limits(response)
                continue
            self._respect_rate_limits(response)
            response.raise_for_status()
            return response

    @staticmethod
    def _respect_rate_limits(response: requests.Response) -> None:
        remaining = response.headers.get("X-Rate-Limit-Remaining")
        reset = response.headers.get("X-Rate-Limit-Reset")
        retry = response.headers.get("Retry-After")
        delay = 0
        if retry and retry.isdigit():
            delay = int(retry)
        elif remaining == "0" and reset and reset.isdigit():
            delay = int(reset)
        if delay > 0:
            time.sleep(delay)

    def _assignment_to_doc(self, item: Dict[str, Any]) -> Document:
        content = f"{item.get('name', '')}\n\n{item.get('description', '')}"
        metadata = self._build_metadata(item, "assignment", item.get("html_url"))
        return Document(page_content=content, metadata=metadata)

    def _quiz_to_doc(self, item: Dict[str, Any]) -> Document:
        content = f"{item.get('title', '')}\n\n{item.get('description', '')}"
        metadata = self._build_metadata(item, "quiz", item.get("html_url"))
        return Document(page_content=content, metadata=metadata)

    def _announcement_to_doc(self, item: Dict[str, Any]) -> Document:
        content = f"{item.get('title', '')}\n\n{item.get('message', '')}"
        metadata = self._build_metadata(item, "announcement", item.get("html_url"))
        return Document(page_content=content, metadata=metadata)

    def _build_metadata(
        self, item: Dict[str, Any], resource_type: str, source: Optional[str]
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "course_id": self.course_id,
            "resource_type": resource_type,
        }
        if source:
            metadata["source"] = source
        if "id" in item:
            metadata["id"] = item["id"]
        timestamp = (
            item.get("updated_at") or item.get("created_at") or item.get("posted_at")
        )
        if timestamp:
            metadata["timestamp"] = timestamp
        return metadata
