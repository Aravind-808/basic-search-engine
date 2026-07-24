'''
search_engine.py
 
it goes DocumentCollector -> Tokenizer -> InvertedIndex, then:
  - TF-IDF per document
  - query processing (same pipeline as documents)
  - cosine similarity ranking
'''
 
import math
import re
 
from file_loader import DocumentCollector
from tokenizer import Tokenizer
from inverted_index import InvertedIndex
from boolean_search import boolean_search, ops
from fuzzy_search import find_closest_terms
 
 
class SearchEngine:
    def __init__(self, fuzzy_max_distance=2):
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex()
        self.processed_docs = {} # doc_id -> {filename, tokens, term_freq}
        self.raw_content = {} # doc_id -> original untokenized text
        self.doc_vectors = {} # doc_id -> {term: tfidf_weight}
        self.doc_norms = {} # doc_id -> vector magnitude (precomputed)
        self.fuzzy_max_distance = fuzzy_max_distance
 
    def load_and_index(self, folder_path):
        collector = DocumentCollector()
        collector.load_files(folder_path)
 
        self.raw_content = {doc["id"]: doc["content"] for doc in collector.file_data}
 
        self.processed_docs = self.tokenizer.process_documents(collector.file_data)
        self.index.build(self.processed_docs)
 
        self.build_doc_vectors()
 
    def build_doc_vectors(self):
        """
        for every document, compute a TF-IDF weight for each of its terms:
            tfidf(term, doc) = term_freq(term, doc) * idf(term)
        store the vector and its precomputed norm (for cosine similarity later).
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
        build a TF-IDF vector for the query the same way as documents:
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
 
    def apply_fuzzy_correction(self, query_terms):
        """
        For any stemmed query term that doesn't exist in the index at all,
        look for the closest known term (edit distance) and substitute it.
        Returns (corrected_terms, corrections) where corrections is a list
        of (original_term, corrected_term) pairs actually applied.
        """
        vocabulary = self.index.vocabulary()
        corrected_terms = []
        corrections = []
 
        for term in query_terms:
            if self.index.document_frequency(term) > 0:
                corrected_terms.append(term)
                continue
 
            candidates = find_closest_terms(
                term, vocabulary, max_distance=self.fuzzy_max_distance, max_results=1
            )
            if candidates:
                corrected_terms.append(candidates[0])
                corrections.append((term, candidates[0]))
            else:
                corrected_terms.append(term)  # keep as-is, will just score 0
 
        return corrected_terms, corrections
 
    def get_snippet(self, doc_id, query_words, window=8):
        """
        returns a short excerpt of the ORIGINAL (untokenized) document text
        around the first occurrence of any raw query word. Falls back to
        the start of the document if no exact word match is found.
        """
        content = self.raw_content.get(doc_id, "")
        words = content.split()
 
        for i, word in enumerate(words):
            cleaned = re.sub(r"[^\w]", "", word).lower()
            if cleaned in query_words:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                excerpt = " ".join(words[start:end])
                return ("..." if start > 0 else "") + excerpt + ("..." if end < len(words) else "")
 
        # fallback: no exact raw-word match (e.g. only the stemmed form matched)
        preview = " ".join(words[:window * 2])
        return preview + ("..." if len(words) > window * 2 else "")
 
    def search(self, query, top_k=5):
        """
        returns a ranked list of (doc_id, filename, score, snippet) tuples,
        highest score first. Docs with score 0 are excluded.
 
        also applies fuzzy correction: if a query term isn't in the index
        at all, the closest known term (by edit distance) is substituted.
        Returns corrections alongside results as (results, corrections).
        """
        raw_query_words = {re.sub(r"[^\w]", "", w).lower() for w in query.split()}
 
        query_terms = self.tokenizer.tokenize(query)
        query_terms, corrections = self.apply_fuzzy_correction(query_terms)
        query_vec = self.query_vector(query_terms)
 
        results = []
        for doc_id, doc_vec in self.doc_vectors.items():
            score = self.cosine_similarity(query_vec, doc_vec, self.doc_norms[doc_id])
            if score > 0:
                filename = self.processed_docs[doc_id]["filename"]
                snippet = self.get_snippet(doc_id, raw_query_words)
                results.append((doc_id, filename, score, snippet))
 
        results.sort(key=lambda r: r[2], reverse=True)
        return results[:top_k], corrections
 
    def is_boolean_query(self, query):
        """true if the query contains AND/OR/NOT ops."""
        tokens = {tok.upper() for tok in query.split()}
        return bool(tokens & ops)
 
    def boolean_search(self, query):
        """
        returns a list of (doc_id, filename, snippet) tuples matching the
        boolean query. No ranking/scoring - a document either matches or
        it doesn't.
        """
        raw_query_words = {
            re.sub(r"[^\w]", "", w).lower()
            for w in query.split()
            if w.upper() not in ops
        }
 
        all_doc_ids = set(self.processed_docs.keys())
        matching_ids = boolean_search(query, self.tokenizer, self.index, all_doc_ids)
 
        results = []
        for doc_id in matching_ids:
            filename = self.processed_docs[doc_id]["filename"]
            snippet = self.get_snippet(doc_id, raw_query_words)
            results.append((doc_id, filename, snippet))
 
        results.sort(key=lambda r: r[0])
        return results
 
 
if __name__ == "__main__":
    engine = SearchEngine()
    engine.load_and_index("sample_corpus")
 
    query = "machine learning python"
    results, corrections = engine.search(query)
 
    print(f"Query: {query}\n")
    for doc_id, filename, score, snippet in results:
        print(f"[{score:.4f}] doc {doc_id} - {filename}")
        print(f"...{snippet}...")