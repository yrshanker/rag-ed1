from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def generate_piazza_export(
    out_path: Path | str,
    *,
    course_number: str = "12345",
    course_name: str = "piazza_sample",
    num_posts: int = 1,
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

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

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

    # By default generate a single example post matching earlier tests. If
    # num_posts > 1, generate a small reply chain (p1 -> p2 -> p3 ...).
    content = []
    if num_posts == 1:
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
    else:
        for i in range(1, num_posts + 1):
            post_id = f"p{i}"
            post = {
                "id": post_id,
                "subject": f"Post {post_id}",
                "content": f"<p>Content for {post_id}</p>",
                "type": "question" if i == 1 else "followup",
                "tag_good_arr": [],
                "created": timestamp,
                "views": 1,
                "score": 0,
                "editors": ["u1"],
                "anonimity": "no",
                "thread_id": "p1",
            }
            if i > 1:
                # reply to previous post
                post["parent_id"] = f"p{i-1}"
            content.append(post)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        config_info = zipfile.ZipInfo("config.json")
        config_info.date_time = (2023, 1, 1, 0, 0, 0)
        zf.writestr(config_info, json.dumps(config))
        users_info = zipfile.ZipInfo("users.json")
        users_info.date_time = (2023, 1, 2, 0, 0, 0)
        zf.writestr(users_info, json.dumps(users))
        content_info = zipfile.ZipInfo("class_content_flat.json")
        content_info.date_time = (2023, 1, 3, 0, 0, 0)
        zf.writestr(content_info, json.dumps(content))

    return path
