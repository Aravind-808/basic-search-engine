'''
btree_index.py

same as the dict version of inverted index, just used in a btree instead to scale 
'''

import math
 
from btree import BTree
 
 
class BTreeInvertedIndex:
    def __init__(self, order=32):
        self.tree = BTree(order=order)
        self.doc_count = 0
 
    def build(self, processed_docs):
        self.doc_count = len(processed_docs)
 
        # gather everything per term first, then insert each term once -
        # way fewer tree operations than inserting per (term, doc) pair
        entries = {}
        for doc_id, data in processed_docs.items():
            for term, freq in data["term_freq"].items():
                entries.setdefault(term, {"freq": {}, "pos": {}})
                entries[term]["freq"][doc_id] = freq
 
            for position, term in enumerate(data["tokens"]):
                entries.setdefault(term, {"freq": {}, "pos": {}})
                entries[term]["pos"].setdefault(doc_id, []).append(position)
 
        for term, entry in entries.items():
            self.tree.insert(term, entry)
 
    def get_entry(self, term):
        entry = self.tree.search(term)
        return entry if entry is not None else {"freq": {}, "pos": {}}
 
    def get_postings(self, term):
        return self.get_entry(term)["freq"]
 
    def get_positions(self, term, doc_id):
        return self.get_entry(term)["pos"].get(doc_id, [])
 
    def document_frequency(self, term):
        return len(self.get_postings(term))
 
    def idf(self, term):
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(self.doc_count / df)
 
    def vocabulary(self):
        return [term for term, entry in self.tree.inorder()]
 
    def phrase_search(self, phrase_terms):
        if not phrase_terms:
            return set()
 
        first_term = phrase_terms[0]
        candidate_docs = set(self.get_entry(first_term)["pos"].keys())
 
        for term in phrase_terms[1:]:
            candidate_docs &= set(self.get_entry(term)["pos"].keys())
 
        matching_docs = set()
        for doc_id in candidate_docs:
            first_positions = self.get_entry(first_term)["pos"][doc_id]
            for start_pos in first_positions:
                if self.sequence_matches(phrase_terms, doc_id, start_pos):
                    matching_docs.add(doc_id)
                    break
 
        return matching_docs
 
    def sequence_matches(self, phrase_terms, doc_id, start_pos):
        for offset, term in enumerate(phrase_terms[1:], start=1):
            expected_pos = start_pos + offset
            term_positions = self.get_entry(term)["pos"].get(doc_id, [])
            if expected_pos not in term_positions:
                return False
        return True
 
    def export_postings(self):
        return {term: entry["freq"] for term, entry in self.tree.inorder()}
 
    def export_positions(self):
        return {term: entry["pos"] for term, entry in self.tree.inorder()}
 
    def load_from_export(self, postings, positions, doc_count):
        
        self.tree = BTree(order=self.tree.order)
        self.doc_count = doc_count
 
        all_terms = set(postings.keys()) | set(positions.keys())
        for term in all_terms:
            entry = {
                "freq": postings.get(term, {}),
                "pos": positions.get(term, {}),
            }
            self.tree.insert(term, entry)
 
 
if __name__ == "__main__":
    from file_loader import DocumentCollector
    from tokenizer import Tokenizer
 
    collector = DocumentCollector()
    collector.load_files("corpus")
    tokenizer = Tokenizer()
    processed = tokenizer.process_documents(collector.file_data)
 
    index = BTreeInvertedIndex()
    index.build(processed)
 
    print("B-tree height:", index.tree.height())
    print("Vocabulary size:", len(index.vocabulary()))
    print()
    print("postings('treatment'):", index.get_postings("treatment"))
    print("idf('treatment'):", round(index.idf("treatment"), 4))

    # idf and posting matches well with inverted_index!