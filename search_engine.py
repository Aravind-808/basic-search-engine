'''
search_engine.py

Ties together DocumentCollector -> Tokenizer -> InvertedIndex, then adds:
  - TF-IDF vector computation per document
  - Query processing (same pipeline as documents)
  - Cosine similarity ranking
  - A single search() entry point
'''

import math

from file_loader import DocumentCollector
from tokenizer import Tokenizer
from inverted_index import InvertedIndex


class SearchEngine:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex()
        self.processed_docs = {}    
        self.doc_vectors = {}       
        self.doc_norms = {}          

    def load_and_index(self, folder_path):
        collector = DocumentCollector()
        collector.load_files(folder_path)

        self.processed_docs = self.tokenizer.process_documents(collector.file_data)
        self.index.build(self.processed_docs)

        self.build_doc_vectors()

    def build_doc_vectors(self):
        """
        For every document, compute a TF-IDF weight for each of its terms:
            tfidf(term, doc) = term_freq(term, doc) * idf(term)
        Store the vector and its precomputed norm (for cosine similarity later).
        """
        for doc_id, data in self.processed_docs.items():
            vector = {}
            for term, freq in data["term_freq"].items():
                vector[term] = freq * self.index.idf(term)
            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = self.magnitude(vector)

    def magnitude(self, vector):
        return math.sqrt(sum(weight ** 2 for weight in vector.values()))

    def query_vector(self, query_terms):
        """
        Build a TF-IDF vector for the query the same way as documents:
        term frequency within the query, times the term's corpus IDF.
        """
        freqs = {}
        for term in query_terms:
            freqs[term] = freqs.get(term, 0) + 1

        vector = {}
        for term, freq in freqs.items():
            vector[term] = freq * self.index.idf(term)
        return vector

    def cosine_similarity(self, query_vec, doc_vec, doc_norm):
        dot_product = 0.0
        for term, q_weight in query_vec.items():
            if term in doc_vec:
                dot_product += q_weight * doc_vec[term]

        query_norm = self.magnitude(query_vec)
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return dot_product / (query_norm * doc_norm)

    def search(self, query, top_k=5):
        """
        Returns a ranked list of (doc_id, filename, score) tuples,
        highest score first. Docs with score 0 are excluded.
        """
        query_terms = self.tokenizer.tokenize(query)
        query_vec = self.query_vector(query_terms)

        results = []
        for doc_id, doc_vec in self.doc_vectors.items():
            score = self.cosine_similarity(query_vec, doc_vec, self.doc_norms[doc_id])
            if score > 0:
                filename = self.processed_docs[doc_id]["filename"]
                results.append((doc_id, filename, score))

        results.sort(key=lambda r: r[2], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    engine = SearchEngine()
    engine.load_and_index("test_docs")

    query = "machine learning python"
    results = engine.search(query)

    print(f"Query: {query}\n")
    for doc_id, filename, score in results:
        print(f"  [{score:.4f}] doc {doc_id} - {filename}")