"""
This file contains the PiazzaLoader class, which is responsible for loading files from Piazza
"""

import datetime
import os
from pathlib import Path

import langchain_community.document_loaders
import langchain_core.document_loaders
import langchain_core.documents
import tqdm


class PiazzaLoader(langchain_core.document_loaders.BaseLoader):
    """
    PiazzaLoader loads files from a zipped file exported from Piazza.

    Parameters
    ----------
    file_path : str
        Path to the zipped Piazza file.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize PiazzaLoader.

        Parameters
        ----------
        file_path : str
            Path to the zipped Piazza file.
        """
        path = Path(file_path)
        if not path.is_file():
            msg = f"Piazza file '{file_path}' does not exist or is not a file."
            raise FileNotFoundError(msg)
        self.zipped_file_path = str(path)
        self.course = path.stem

    def load(self) -> list[langchain_core.documents.Document]:
        """
        Load all files from the zipped Piazza export.

        Returns
        -------
        list of Document
            List of loaded documents.

        Examples
        --------
        >>> loader = PiazzaLoader('piazza.zip')
        >>> docs = loader.load()
        >>> print(len(docs))
        """
        from rag_ed.loaders.utils import extract_zip_to_temp

        def process(temp_dir: str) -> list[langchain_core.documents.Document]:
            file_paths = []
            for root, _, files in tqdm.tqdm(os.walk(temp_dir)):
                for file in files:
                    file_paths.append(os.path.join(root, file))
            return self._load_files(file_paths)

        return extract_zip_to_temp(self.zipped_file_path, process)

    # _unzip_piazza_file is no longer needed; all processing is done in load()

    def _load_files(
        self, list_of_files_to_load: list[str]
    ) -> list[langchain_core.documents.Document]:
        """
        Load files from a list of file paths.

        Parameters
        ----------
        list_of_files_to_load : list of str
            List of file paths to load.

        Returns
        -------
        list of Document
            List of loaded documents.

        Examples
        --------
        >>> loader = PiazzaLoader('piazza.zip')
        >>> files = loader._unzip_piazza_file()
        >>> docs = loader._load_files(files)
        """
        loaded_documents = []
        for file_path in tqdm.tqdm(list_of_files_to_load):
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".csv":
                    new_documents = langchain_community.document_loaders.CSVLoader(
                        file_path
                    ).load()
                elif file_extension == ".json":
                    new_documents = langchain_community.document_loaders.JSONLoader(
                        file_path, jq_schema=".", text_content=False
                    ).load()
                else:
                    continue  # Skip other file types

                timestamp = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).isoformat()
                for doc in new_documents:
                    doc.metadata.setdefault("source", file_path)
                    doc.metadata["course"] = self.course
                    doc.metadata["timestamp"] = timestamp
                loaded_documents += new_documents
        return loaded_documents


if __name__ == "__main__":
    # Example usage
    loader = PiazzaLoader("/Users/work/Downloads/mech2-piazza.zip")
    documents = loader.load()
    for doc in documents:
        print(doc)
