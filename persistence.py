'''
persistence.py

IndexCache: saves/loads a SearchEngine's built index to/from a JSON file,
so a folder doesn't get re-tokenized on every run.

wat this class solves:
  JSON only allows string dict keys, but doc_ids are ints internally,
     so keys need converting both ways on save/load.
    there needs to be a cheap way to tell whether the source folder has
     changed since the cache was written - a "fingerprint" of
     filename+mtime+size for every .txt file, sorted for stable comparison.
'''

import json
import os


class IndexCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path

    # thank u claude for the fingerprinting method
    def compute_fingerprint(self, folder_path):
        """
        Returns a sorted list of [abs_path, mtime, size] for every .txt file
        under folder_path. Two calls return equal lists iff nothing changed
        (no added/removed/modified files).
        """
        entries = []
        for root, _folders, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    full_path = os.path.abspath(os.path.join(root, file))
                    stat = os.stat(full_path)
                    entries.append([full_path, stat.st_mtime, stat.st_size])

        entries.sort(key=lambda e: e[0])
        return entries

    def stringify_outer_keys(self, d):
        # {doc_id: value} -> {"doc_id": value}
        return {str(k): v for k, v in d.items()}

    def intify_outer_keys(self, d):
        # {"doc_id": value} -> {doc_id: value}
        return {int(k): v for k, v in d.items()}

    def stringify_inner_keys(self, d):
        # {term: {doc_id: value}} -> {term: {"doc_id": value}}
        return {term: {str(k): v for k, v in inner.items()} for term, inner in d.items()}

    def intify_inner_keys(self, d):
        # {term: {"doc_id": value}} -> {term: {doc_id: value}}
        return {term: {int(k): v for k, v in inner.items()} for term, inner in d.items()}

    def load(self):
        """
        returns the raw cached dict exactly as stored in JSON (string keys,
        untouched), or None if the cache file doesn't exist or is corrupt.
        """
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def is_fresh(self, cached_data, folder_path):
        # True if cached_data exists and matches the folder's current fingerprint.
        if cached_data is None:
            return False
        return cached_data.get("fingerprint") == self.compute_fingerprint(folder_path)

    def to_engine_state(self, cached_data):
        # Converts a raw cached dict (string keys, JSON-shaped) back into
        # the python objects a SearchEngine expects (int doc_id keys).
        
        return {
            "doc_count": cached_data["doc_count"],
            "index": self.intify_inner_keys(cached_data["index"]),
            "positions": self.intify_inner_keys(cached_data["positions"]),
            "processed_docs": self.intify_outer_keys(cached_data["processed_docs"]),
            "raw_content": self.intify_outer_keys(cached_data["raw_content"]),
            "doc_vectors": self.intify_outer_keys(cached_data["doc_vectors"]),
            "doc_norms": self.intify_outer_keys(cached_data["doc_norms"]),
            "doc_lengths": self.intify_outer_keys(cached_data["doc_lengths"]),
            "avg_doc_length": cached_data["avg_doc_length"],
        }

    def save(self, folder_path, engine_state):
        """
        engine_state: dict with keys doc_count, index, positions,
        processed_docs, raw_content, doc_vectors, doc_norms, doc_lengths,
        avg_doc_length (all doc_id-keyed items using int keys, as
        SearchEngine holds them internally).
        
        writes them to self.cache_path as JSON, alongside a fingerprint
        of folder_path used to detect staleness later.
        """
        data = {
            "fingerprint": self.compute_fingerprint(folder_path),
            "doc_count": engine_state["doc_count"],
            "index": self.stringify_inner_keys(engine_state["index"]),
            "positions": self.stringify_inner_keys(engine_state["positions"]),
            "processed_docs": self.stringify_outer_keys(engine_state["processed_docs"]),
            "raw_content": self.stringify_outer_keys(engine_state["raw_content"]),
            "doc_vectors": self.stringify_outer_keys(engine_state["doc_vectors"]),
            "doc_norms": self.stringify_outer_keys(engine_state["doc_norms"]),
            "doc_lengths": self.stringify_outer_keys(engine_state["doc_lengths"]),
            "avg_doc_length": engine_state["avg_doc_length"],
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)