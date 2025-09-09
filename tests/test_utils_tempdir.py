import os
import tempfile
import zipfile
from rag_ed.loaders.utils import extract_zip_to_temp


def test_extract_zip_to_temp_removes_temp_dir():
    # Create a dummy zip file with a temp file inside
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_file_path = os.path.join(temp_dir, "dummy.txt")
        with open(dummy_file_path, "w") as f:
            f.write("hello")
        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(dummy_file_path, arcname="dummy.txt")

        # Call the extraction function with a process callback
        def process(temp_dir):
            file_paths = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_paths.append(os.path.join(root, file))
            # All returned files should exist inside the context
            for path in file_paths:
                assert os.path.exists(path)
            return file_paths

        file_paths = extract_zip_to_temp(zip_path, process)
        # After extraction, temp dir should be removed
        for path in file_paths:
            assert not os.path.exists(path)
