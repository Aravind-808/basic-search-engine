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

        # first pass: gather term -> {doc_id: freq}
        # then insert each term once as a single B-tree entry
        postings = {}
        for doc_id, data in processed_docs.items():
            for term, freq in data["term_freq"].items():
                postings.setdefault(term, {})[doc_id] = freq

        for term, doc_freqs in postings.items():
            self.tree.insert(term, doc_freqs)

    def get_postings(self, term):
        result = self.tree.search(term)
        return result if result is not None else {}

    def document_frequency(self, term):
        return len(self.get_postings(term))

    def idf(self, term):
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(self.doc_count / df)

    def vocabulary(self):
        return [term for term, postings in self.tree.inorder()]


if __name__ == "__main__":
    from file_loader import DocumentCollector
    from tokenizer import Tokenizer
    from inverted_index import InvertedIndex

    collector = DocumentCollector()
    collector.load_files("topic_corpus")
    tokenizer = Tokenizer()
    processed = tokenizer.process_documents(collector.file_data)

    dict_index = InvertedIndex()
    dict_index.build(processed)

    btree_index = BTreeInvertedIndex()
    btree_index.build(processed)

    print("B-tree height:", btree_index.tree.height())
    print("Vocabulary sizes match:", len(dict_index.vocabulary()) == len(btree_index.vocabulary()))

    # correctness check: every term's postings and idf must match exactly
    mismatches = 0
    for term in dict_index.vocabulary():
        if dict_index.get_postings(term) != btree_index.get_postings(term):
            mismatches += 1
            continue
        if abs(dict_index.idf(term) - btree_index.idf(term)) > 1e-9:
            mismatches += 1

    print(f"mismatches vs dict-based index: {mismatches} (should be 0)")