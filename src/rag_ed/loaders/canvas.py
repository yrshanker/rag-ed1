"""
This file contains the CanvasLoader class, which is responsible for loading files from Canvas
"""

import datetime
import os

import langchain_community.document_loaders
import langchain_core.document_loaders
import langchain_core.documents
import tqdm


class CanvasLoader(langchain_core.document_loaders.BaseLoader):
    """
    The CanvasLoader class is responsible for loading files from a zipped .imscc file which can be exported from Canvas.
    """

    def __init__(self, file_path: str):
        """
        Initialize the CanvasLoader with the path to the zipped .imscc file.

        Args:
            file_path (str): The path to the zipped .imscc file.
        """
        if not os.path.exists(file_path):
            msg = f"Canvas file '{file_path}' does not exist."
            raise FileNotFoundError(msg)
        self.zipped_file_path = file_path
        self.course = os.path.splitext(os.path.basename(file_path))[0]

    def load(self):
        """
        Load the files from the zipped .imscc file.
        Returns:
            list: A list of loaded documents.
        """
        list_of_files_to_load = self._unzip_imscc_file()
        return self._load_files(list_of_files_to_load)

    def _unzip_imscc_file(self) -> list[str]:
        """
        Unzip the .imscc file to a temporary location, recursively traverse the file tree of the unzipped directory,
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
                if file_extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                    continue  # Skip image files
                if file_extension == ".html":
                    new_documents = (
                        langchain_community.document_loaders.UnstructuredHTMLLoader(
                            file_path
                        ).load()
                    )
                elif file_extension == ".xml":
                    new_documents = (
                        langchain_community.document_loaders.UnstructuredXMLLoader(
                            file_path
                        ).load()
                    )
                elif file_extension == ".pdf":
                    new_documents = (
                        langchain_community.document_loaders.UnstructuredPDFLoader(
                            file_path
                        ).load()
                    )
                else:
                    with open(file_path, "r") as file:
                        content = file.read()
                    new_documents = [
                        langchain_core.documents.Document(
                            page_content=content, metadata={"source": file_path}
                        )
                    ]

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
    loader = CanvasLoader(
        "/Users/work/Downloads/special-topics-designing-and-deploying-ai-slash-ml-systems-export.imscc"
    )
    documents = loader.load()
    for doc in documents:
        print(doc)
