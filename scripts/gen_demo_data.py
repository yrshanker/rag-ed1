"""Generate deterministic tiny demo datasets (Canvas IMSCC + Piazza ZIP).

Usage (from repo root):

    PYTHONPATH=src python scripts/gen_demo_data.py --out-dir demo_data

This script produces two files inside the output directory:

    demo_data/demo_course.imscc
    demo_data/demo_piazza.zip

They are created using the existing test utility generators so they are
small, dependency-light, and deterministic. Use these artifacts for quick
manual graph builds or local smoke tests:

    from rag_ed.graphs import graph_from_canvas, graph_from_piazza
    g_canvas = graph_from_canvas("demo_data/demo_course.imscc")
    g_piazza = graph_from_piazza("demo_data/demo_piazza.zip")

The generators embed fixed timestamps (zip entry metadata) ensuring
consistent chronological ordering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.imscc_utils import generate_imscc
from tests.piazza_utils import generate_piazza_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo IMSCC + Piazza archives")
    parser.add_argument(
        "--out-dir", default="demo_data", help="Directory to write generated archives"
    )
    parser.add_argument(
        "--canvas-name", default="demo_course", help="Base name for Canvas export (.imscc)"
    )
    parser.add_argument(
        "--piazza-name", default="demo_piazza", help="Base name for Piazza export (.zip)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_path = out_dir / f"{args.canvas_name}.imscc"
    piazza_path = out_dir / f"{args.piazza_name}.zip"

    generate_imscc(canvas_path)
    generate_piazza_export(piazza_path)

    print(f"Generated: {canvas_path}")
    print(f"Generated: {piazza_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
