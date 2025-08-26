from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path


def generate_piazza_export(
    out_path: Path | str,
    *,
    course_number: str = "12345",
    course_name: str = "piazza_sample",
) -> Path:
    """Generate a small Piazza export archive.

    Parameters
    ----------
    out_path : Path | str
        Destination ``.zip`` path.
    course_number : str, optional
        Course identifier written to ``config.json``.
    course_name : str, optional
        Human-readable course name.

    Returns
    -------
    Path
        Path to the generated ``.zip`` file.
    """
    path = Path(out_path).with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    config = {
        "course_number": course_number,
        "name": course_name,
        "lti_ids": [],
        "access_code_on": False,
        "inst_self_signup_on": False,
    }

    users = [
        {
            "user_id": "u1",
            "name": "Example Student",
            "email": "student@example.com",
            "lti_ids": ["lti1"],
            "days": 1,
            "posts": 1,
            "asks": 1,
            "answers": 0,
            "views": 1,
        }
    ]

    content = [
        {
            "id": "p1",
            "subject": "Hello from Piazza",
            "content": "<p>Hello from Piazza</p>",
            "type": "question",
            "tag_good_arr": [],
            "created": timestamp,
            "views": 1,
            "score": 0,
            "editors": ["u1"],
            "anonimity": "no",
            "thread_id": "p1",
        }
    ]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(config))
        zf.writestr("users.json", json.dumps(users))
        zf.writestr("class_content_flat.json", json.dumps(content))

    return path
