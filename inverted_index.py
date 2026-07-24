'''
inverted_index.py

Builds an inverted index from tokenized documents (output of
Tokenizer.process_documents), and computes IDF for TF-IDF scoring.
'''

import math


class InvertedIndex:
    def __init__(self):
        self.index = {}       
        self.doc_count = 0    

    def build(self, processed_docs):
        """
        processed_docs: output of Tokenizer.process_documents()
        {
            doc_id: {"filename":..., "tokens":[...], "term_freq": {...}}
        }
        """
        self.doc_count = len(processed_docs)
        for doc_id, data in processed_docs.items():
            for term, freq in data["term_freq"].items():
                self.index.setdefault(term, {})[doc_id] = freq

    def get_postings(self, term):
        """Returns {doc_id: frequency} for a term, or {} if not found."""
        return self.index.get(term, {})

    def document_frequency(self, term):
        """Number of documents containing this term."""
        return len(self.index.get(term, {}))

    def idf(self, term):
        """
        Inverse document frequency for a term.
        Smoothed to avoid divide-by-zero if a term is somehow absent.
        """
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(self.doc_count / df)

    def vocabulary(self):
        """All unique terms in the index."""
        return list(self.index.keys())


if __name__ == "__main__":
    from file_loader import DocumentCollector
    from tokenizer import Tokenizer

    collector = DocumentCollector()
    collector.load_files("test_docs")

    tokenizer = Tokenizer()
    processed = tokenizer.process_documents(collector.file_data)

    index = InvertedIndex()
    index.build(processed)

    print("Vocabulary size:", len(index.vocabulary()))
    for term in ["machin", "learn", "python"]:
        print(f"{term:10} postings={index.get_postings(term)}  df={index.document_frequency(term)}  idf={index.idf(term):.4f}")