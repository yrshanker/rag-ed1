import sys
from pathlib import Path
import zipfile
import json


if __name__ == "__main__":
    # Ensure repo root is on sys.path so 'tests' package and 'rag_ed' resolve.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from tests.piazza_utils import generate_piazza_export
    from rag_ed.loaders.piazza import PiazzaLoader

    p = Path("tmp_piazza.zip")
    print("generating")
    generate_piazza_export(p, num_posts=3)
    print("jsonloader raw:")

    with zipfile.ZipFile(p) as zf:
        name = None
        for info in zf.infolist():
            if info.filename.endswith("class_content_flat.json"):
                name = info.filename
                break
        data = zf.read(name)
        obj = json.loads(data)
        print(type(obj), len(obj))
        for item in obj:
            print("item keys:", list(item.keys()))

    print("\nPiazzaLoader.load() output:")
    pl = PiazzaLoader(str(p))
    docs = pl.load()
    print("docs:", len(docs))
    for d in docs:
        print("page_content_type:", type(d.page_content), "metadata:", d.metadata)
