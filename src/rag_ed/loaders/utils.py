import tempfile
import zipfile
from typing import Callable, TypeVar

T = TypeVar("T")


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
