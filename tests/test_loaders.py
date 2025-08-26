from pathlib import Path

from rag_ed.loaders.canvas import CanvasLoader
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


def test_piazza_loader_returns_document(tmp_path: Path) -> None:
    path = generate_piazza_export(tmp_path / "piazza_sample.zip")
    docs = PiazzaLoader(str(path)).load()
    assert len(docs) == 3
    assert any("Hello from Piazza" in d.page_content for d in docs)
    for doc in docs:
        assert doc.metadata["course"] == "piazza_sample"
