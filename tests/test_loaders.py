from pathlib import Path
import datetime
import time
from typing import Iterable

import pytest

from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.canvas_api import CanvasAPILoader
from rag_ed.loaders.piazza import PiazzaLoader
from rag_ed.loaders.piazza_api import PiazzaAPILoader
from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export
from rag_ed.loaders.utils import extract_zip


def test_canvas_loader_returns_document(tmp_path: Path) -> None:
    path = generate_imscc(tmp_path / "canvas_sample.imscc", title="canvas_sample")
    docs = CanvasLoader(str(path)).load()
    assert len(docs) == 3
    assert any("minimal Common Cartridge web page" in d.page_content for d in docs)
    for doc in docs:
        assert doc.metadata["course"] == "canvas_sample"
        datetime.datetime.fromisoformat(doc.metadata["timestamp"])


def test_canvas_loader_missing_file() -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Canvas file 'does_not_exist.imscc' does not exist or is not a file.",
    ):
        CanvasLoader("does_not_exist.imscc")


def test_canvas_api_loader_fetches_documents() -> None:
    base_url = "https://canvas.example"
    course_id = 123
    token = "tok"

    import responses

    with responses.RequestsMock() as rsps:
        # assignments pagination
        first_assign_url = f"{base_url}/api/v1/courses/{course_id}/assignments"
        second_assign_url = f"{first_assign_url}?page=2"
        rsps.add(
            "GET",
            first_assign_url,
            json=[
                {
                    "id": 1,
                    "name": "A1",
                    "description": "desc1",
                    "html_url": "a1",
                    "updated_at": "2023-01-01T00:00:00Z",
                }
            ],
            adding_headers={"Link": f'<{second_assign_url}>; rel="next"'},
        )
        rsps.add(
            "GET",
            second_assign_url,
            json=[
                {
                    "id": 2,
                    "name": "A2",
                    "description": "desc2",
                    "html_url": "a2",
                    "updated_at": "2023-01-02T00:00:00Z",
                }
            ],
        )
        # quizzes
        rsps.add(
            "GET",
            f"{base_url}/api/v1/courses/{course_id}/quizzes",
            json=[
                {
                    "id": 10,
                    "title": "Q1",
                    "description": "qdesc",
                    "html_url": "q1",
                    "updated_at": "2023-01-03T00:00:00Z",
                }
            ],
        )
        # announcements
        rsps.add(
            "GET",
            f"{base_url}/api/v1/announcements",
            match=[
                responses.matchers.query_param_matcher(
                    {"context_codes[]": f"course_{course_id}"}
                )
            ],
            json=[
                {
                    "id": 100,
                    "title": "Ann",
                    "message": "announcement",
                    "html_url": "ann",
                    "posted_at": "2023-01-04T00:00:00Z",
                }
            ],
        )

        loader = CanvasAPILoader(base_url, course_id, token)
        docs = loader.load()

    assert len(docs) == 4
    types = {d.metadata["resource_type"] for d in docs}
    assert types == {"assignment", "quiz", "announcement"}


def test_canvas_api_loader_respects_rate_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://canvas.example"
    course_id = 1
    token = "tok"

    import responses

    called: list[int] = []
    monkeypatch.setattr(time, "sleep", lambda s: called.append(s))

    with responses.RequestsMock() as rsps:
        rsps.add(
            "GET",
            f"{base_url}/api/v1/courses/{course_id}/assignments",
            json=[],
            adding_headers={
                "X-Rate-Limit-Remaining": "0",
                "X-Rate-Limit-Reset": "1",
            },
        )
        rsps.add(
            "GET",
            f"{base_url}/api/v1/courses/{course_id}/quizzes",
            json=[],
        )
        rsps.add(
            "GET",
            f"{base_url}/api/v1/announcements",
            match=[
                responses.matchers.query_param_matcher(
                    {"context_codes[]": f"course_{course_id}"}
                )
            ],
            json=[],
        )

        loader = CanvasAPILoader(base_url, course_id, token)
        loader.load()

    assert called == [1]


def test_extract_zip_lists_files(tmp_path: Path) -> None:
    canvas = generate_imscc(tmp_path / "course.imscc")
    piazza = generate_piazza_export(tmp_path / "piazza.zip")
    canvas_files = extract_zip(str(canvas))
    piazza_files = extract_zip(str(piazza))
    assert any(p.endswith("imsmanifest.xml") for p in canvas_files)
    assert any(p.endswith("config.json") for p in piazza_files)


def test_piazza_loader_returns_document(tmp_path: Path) -> None:
    path = generate_piazza_export(tmp_path / "piazza_sample.zip")
    docs = PiazzaLoader(str(path)).load()
    # PiazzaLoader may return the class_content_flat posts plus extra files
    # (config.json, users.json). Ensure at least the three posts exist.
    assert len(docs) >= 3
    assert any("Hello from Piazza" in d.page_content for d in docs)
    for doc in docs:
        assert doc.metadata["course"] == "piazza_sample"
        datetime.datetime.fromisoformat(doc.metadata["timestamp"])


def test_piazza_loader_missing_file() -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Piazza file 'does_not_exist.zip' does not exist or is not a file.",
    ):
        PiazzaLoader("does_not_exist.zip")


def test_piazza_api_loader_fetches_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = [
        {
            "id": "p1",
            "nr": 1,
            "history": [
                {
                    "subject": "Hello",
                    "content": "<p>World</p>",
                    "created": "2024-01-01T00:00:00Z",
                }
            ],
        },
        {
            "id": "p2",
            "nr": 2,
            "history": [
                {
                    "subject": "Another",
                    "content": "<p>Post</p>",
                    "created": "2024-01-02T00:00:00Z",
                }
            ],
        },
    ]

    class FakeNetwork:
        def iter_all_posts(self) -> Iterable[dict[str, object]]:
            return iter(posts)

    class FakePiazza:
        def user_login(self, *, email: str, password: str) -> None:
            assert email == "e"
            assert password == "p"

        def network(self, network_id: str) -> FakeNetwork:
            assert network_id == "nid"
            return FakeNetwork()

    monkeypatch.setattr("rag_ed.loaders.piazza_api.Piazza", FakePiazza)

    loader = PiazzaAPILoader("nid", email="e", password="p")
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "https://piazza.com/class/nid/post/1"
    assert "Hello" in docs[0].page_content
