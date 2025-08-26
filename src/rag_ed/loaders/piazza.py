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
    The PiazzaLoader class is responsible for loading files from a zipped file which can be exported from Piazza.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the PiazzaLoader with the path to the zipped file.

        Args:
            file_path (str): The path to the zipped file.
        """
        path = Path(file_path)
        if not path.is_file():
            msg = f"Piazza file '{file_path}' does not exist or is not a file."
            raise FileNotFoundError(msg)
        self.zipped_file_path = str(path)
        self.course = path.stem

    def load(self) -> list[langchain_core.documents.Document]:
        """Load the files from the zipped Piazza export.

        Returns:
            list[Document]: A list of loaded documents.
        """
        list_of_files_to_load = self._unzip_piazza_file()
        return self._load_files(list_of_files_to_load)

    def _unzip_piazza_file(self) -> list[str]:
        """
        Unzip the file to a temporary location, recursively traverse the file tree of the unzipped directory,
        and return a list of files to load.

        Returns:
            list: A list of file paths to load.
        """
        import zipfile
        import tempfile

        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(self.zipped_file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        file_paths = []
        for root, _, files in tqdm.tqdm(os.walk(temp_dir)):
            for file in files:
                file_paths.append(os.path.join(root, file))

        return file_paths

    def _load_files(
        self, list_of_files_to_load: list[str]
    ) -> list[langchain_core.documents.Document]:
        """
        Load the files from the list of files to load.

        Args:
            list_of_files_to_load (list): A list of file paths to load.

        Returns:
            list[Document]: A list of loaded documents.
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
