from pathlib import Path
import datetime
import time

import pytest

from rag_ed.loaders.canvas import CanvasLoader
from rag_ed.loaders.canvas_api import CanvasAPILoader
from rag_ed.loaders.piazza import PiazzaLoader
from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export


def test_canvas_loader_returns_document(tmp_path: Path) -> None:
    path = generate_imscc(tmp_path / "canvas_sample.imscc", title="canvas_sample")
    docs = CanvasLoader(str(path)).load()
    assert len(docs) == 2
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


def test_piazza_loader_returns_document(tmp_path: Path) -> None:
    path = generate_piazza_export(tmp_path / "piazza_sample.zip")
    docs = PiazzaLoader(str(path)).load()
    assert len(docs) == 3
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
