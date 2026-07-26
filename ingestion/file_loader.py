'''
file_loader.py
file loader for the search engine:

lets say we have 2 files:
    1) sample1.txt
    2) sample2.txt

the file loader will read this content recursively and return the content as a
list of dicts like so:
[
    {
        "id": 1,
        "filename": "python.txt",
        "content": "Python is awesome..."
    },
    {
        "id": 2,
        "filename": "ml.txt",
        "content": "Machine learning..."
    }
]

this loader will then be passed on to the tokenizer :)
'''
import os
import logging
from ingestion.errors import DuplicateContentError

class DocumentCollector:
    def __init__(self):
        self.file_data = []
        self.file_id = 0
        self.indexed_files = set()

    def load_file(self, file_path):
        # load file and add it to the file_data list
        file_path = os.path.abspath(file_path)
        if file_path in self.indexed_files:
            raise DuplicateContentError("File has already been indexed")
    
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_name = os.path.basename(file_path)
                self.file_id+=1
                self.file_data.append({
                    "id": self.file_id,
                    "filename": file_name,
                    "content": content,
                    "path": file_path
                })
                self.indexed_files.add(file_path)

        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")

    def load_files(self, folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    try:
                        self.load_file(full_path)
                    except DuplicateContentError as e:
                        logging.warning(f"Skipping duplicate: {full_path}")
        return len(self.file_data)




if __name__ == "__main__":
    for root, folders, files in os.walk(r"C:\Users\admin\agent_folder"):
        print(f"root:   {root}")
        print(f"folders {folders}")
        print(f"files   {files}")