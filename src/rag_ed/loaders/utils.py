import os
import tempfile
import zipfile
from typing import Callable, TypeVar, List

T = TypeVar("T")


def extract_zip(path: str) -> List[str]:
    """Extract ``path`` and return absolute paths of contained files.

    Parameters
    ----------
    path:
        Filesystem path to a ``.zip`` archive.

    Returns
    -------
    list[str]
        Paths to all files contained in the archive. The archive is extracted to
        a temporary directory, which is not automatically cleaned up.

    Notes
    -----
    This function intentionally does not clean up the temporary directory so
    that callers can read the extracted files after this function returns.
    Prefer using :func:`extract_zip_to_temp` for lifecycle-safe processing when
    you can perform all I/O within a callback.
    """
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(temp_dir)
    file_paths: list[str] = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def extract_zip_to_temp(zip_path: str, process: Callable[[str], T]) -> T:
    """
    Extracts a zip file to a temporary directory and processes files within the context.

    Parameters
    ----------
    zip_path : str
        Path to the zip file to extract.
    process : Callable[[str], T]
        Function that takes the temp directory path and returns a result.

    Returns
    -------
    T
        Result of processing files in the temp directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        return process(temp_dir)
