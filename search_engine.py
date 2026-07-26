'''
search_engine.py

goes DocumentCollector -> Tokenizer -> InvertedIndex, then does the actual
searching on top of that. two ranking modes exist here: plain TF-IDF +
cosine similarity (search()), and BM25 (search_bm25()). main.py actually
uses BM25 by default, since TF-IDF has two blind spots BM25 fixes - it
doesn't account for document length, and it rewards repeated terms with
basically no limit. BM25 saturates term frequency and normalizes for
doc length instead.

on top of ranking, this also handles boolean search (AND/OR/NOT), exact
phrase search, fuzzy correction for typos, and optional semantic search
using sentence-transformers embeddings if you want to go beyond keyword
matching entirely.

note that using the semantic search option
entirely ignores everything else that is built here
'''

import math
import re

from file_loader import DocumentCollector
from tokenizer import Tokenizer
from inverted_index import InvertedIndex
from boolean_search import boolean_search, ops
from fuzzy_search import find_closest_terms
from persistence import IndexCache


class SearchEngine:
    def __init__(self, fuzzy_max_distance=2):
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex()
        self.processed_docs = {} # doc_id -> {filename, tokens, term_freq}
        self.raw_content = {} # doc_id -> original untokenized text
        self.doc_vectors = {} # doc_id -> {term: tfidf_weight}
        self.doc_norms = {} # doc_id -> vector magnitude, precomputed so cosine similarity is cheap later
        self.doc_lengths = {} # doc_id -> number of tokens, needed for BM25's length normalization
        self.avg_doc_length = 0.0  # avg doc length across the corpus, also needed for BM25
        self.fuzzy_max_distance = fuzzy_max_distance
        self.loaded_from_cache = False  # load_and_index flips this so the caller knows what happened

    def load_and_index(self, folder_path, cache_path=None):
        # if theres a valid up to date cache (folder fingerprint matches),
        # just load everything from disk instead of retokenizing the whole
        # folder again. otherwise rebuild from scratch and save a fresh
        # cache if a cache_path was given
        self.loaded_from_cache = False
        cache = IndexCache(cache_path) if cache_path else None

        if cache:
            cached_data = cache.load()
            if cache.is_fresh(cached_data, folder_path):
                self.apply_cached_state(cache.to_engine_state(cached_data))
                self.loaded_from_cache = True
                return

        collector = DocumentCollector()
        collector.load_files(folder_path)

        self.raw_content = {doc["id"]: doc["content"] for doc in collector.file_data}

        self.processed_docs = self.tokenizer.process_documents(collector.file_data)
        self.index.build(self.processed_docs)

        self.build_doc_vectors()
        self.compute_doc_lengths()

        if cache:
            cache.save(folder_path, self.export_state())

    def apply_cached_state(self, state):
        # just dumps a previously cached state back into the engine's attributes
        self.index.doc_count = state["doc_count"]
        self.index.index = state["index"]
        self.index.positions = state["positions"]
        self.processed_docs = state["processed_docs"]
        self.raw_content = state["raw_content"]
        self.doc_vectors = state["doc_vectors"]
        self.doc_norms = state["doc_norms"]
        self.doc_lengths = state["doc_lengths"]
        self.avg_doc_length = state["avg_doc_length"]

    def export_state(self):
        # the flip side of apply_cached_state, everything IndexCache needs to persist
        return {
            "doc_count": self.index.doc_count,
            "index": self.index.index,
            "positions": self.index.positions,
            "processed_docs": self.processed_docs,
            "raw_content": self.raw_content,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
        }

    def build_doc_vectors(self):
        # for every doc, weight each of its terms by tfidf = term_freq * idf
        # also precompute the vector's magnitude here so cosine_similarity
        # doesnt have to recompute it on every single query later
        for doc_id, data in self.processed_docs.items():
            vector = {}
            for term, freq in data["term_freq"].items():
                vector[term] = freq * self.index.idf(term)
            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = self.magnitude(vector)

    def magnitude(self, vector):
        return math.sqrt(sum(weight ** 2 for weight in vector.values()))

    def query_vector(self, query_terms):
        # build a tfidf vector for the query the exact same way as documents -
        # term frequency within the query itself, times the term's corpus idf
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
        # for any stemmed query term that doesnt exist in the index at all,
        # find the closest known term by edit distance and swap it in.
        # returns the corrected terms plus a list of (original, corrected)
        # pairs so the caller can show the user what got auto-corrected
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
                corrected_terms.append(term) # keep as-is, will just score 0

        return corrected_terms, corrections

    def get_snippet(self, doc_id, query_words, window=8):
        # grabs a short excerpt of the ORIGINAL untokenized text around
        # wherever a raw query word actually shows up, so results look like
        # a real search result preview instead of just a filename + score.
        # falls back to the start of the doc if nothing matches exactly
        content = self.raw_content.get(doc_id, "")
        words = content.split()

        for i, word in enumerate(words):
            cleaned = re.sub(r"[^\w]", "", word).lower()
            if cleaned in query_words:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                excerpt = " ".join(words[start:end])
                return ("..." if start > 0 else "") + excerpt + ("..." if end < len(words) else "")

        # no exact raw-word match (e.g. only the stemmed form matched) - just preview the start
        preview = " ".join(words[:window * 2])
        return preview + ("..." if len(words) > window * 2 else "")

    def search(self, query, top_k=5):
        # plain tfidf/cosine ranked search. returns (results, corrections)
        # where results is a list of (doc_id, filename, score, snippet)
        # tuples, highest score first, docs scoring 0 excluded entirely
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

    def compute_doc_lengths(self):
        # records how many tokens each doc has, plus the corpus wide average -
        # both are needed for bm25's length normalization term
        self.doc_lengths = {
            doc_id: len(data["tokens"]) for doc_id, data in self.processed_docs.items()
        }
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0.0

    def bm25_idf(self, term):
        # bm25 uses a smoother idf than plain log(N/df) - it stays positive
        # even when a term shows up in literally every document, which
        # plain tfidf idf would collapse to zero for
        df = self.index.document_frequency(term)
        n = self.index.doc_count
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, doc_id, query_terms, k1=1.5, b=0.75):
        # sums bm25's per term contribution over every unique query term:
        #   idf(t) * [f(t,D) * (k1+1)] / [f(t,D) + k1 * (1 - b + b*|D|/avgdl)]
        # k1 controls how fast term frequency saturates, b controls how
        # hard doc length gets penalized
        if self.avg_doc_length == 0:
            return 0.0

        doc_term_freqs = self.processed_docs[doc_id]["term_freq"]
        doc_length = self.doc_lengths[doc_id]

        score = 0.0
        for term in set(query_terms):
            freq = doc_term_freqs.get(term, 0)
            if freq == 0:
                continue

            idf = self.bm25_idf(term)
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score

    def search_bm25(self, query, top_k=5, k1=1.5, b=0.75):
        # same interface/return shape as search(), just ranks with bm25
        # instead of tfidf/cosine. this is what main.py actually calls
        raw_query_words = {re.sub(r"[^\w]", "", w).lower() for w in query.split()}

        query_terms = self.tokenizer.tokenize(query)
        query_terms, corrections = self.apply_fuzzy_correction(query_terms)

        results = []
        for doc_id in self.processed_docs:
            score = self.bm25_score(doc_id, query_terms, k1=k1, b=b)
            if score > 0:
                filename = self.processed_docs[doc_id]["filename"]
                snippet = self.get_snippet(doc_id, raw_query_words)
                results.append((doc_id, filename, score, snippet))

        results.sort(key=lambda r: r[2], reverse=True)
        return results[:top_k], corrections

    def build_embeddings(self, model_name="all-MiniLM-L6-v2"):
        # optional, only needed if you want semantic_search() to work.
        # requires pip install sentence-transformers - imported lazily here
        # so nobody's forced to have that dependency just to use bm25/tfidf
        from embedding_search import EmbeddingSearch
        self.embedding_engine = EmbeddingSearch(model_name=model_name)
        self.embedding_engine.build(self.raw_content)

    def semantic_search(self, query, top_k=5):
        # ranks by embedding cosine similarity (meaning) instead of keyword
        # overlap. call build_embeddings() once before using this
        if not hasattr(self, "embedding_engine"):
            raise RuntimeError("Call build_embeddings() before semantic_search().")

        raw_results = self.embedding_engine.search(query, top_k=top_k)
        raw_query_words = {re.sub(r"[^\w]", "", w).lower() for w in query.split()}

        results = []
        for doc_id, score in raw_results:
            filename = self.processed_docs[doc_id]["filename"]
            snippet = self.get_snippet(doc_id, raw_query_words)
            results.append((doc_id, filename, score, snippet))
        return results

    def is_boolean_query(self, query):
        # true if the query contains AND/OR/NOT anywhere in it
        tokens = {tok.upper() for tok in query.split()}
        return bool(tokens & ops)

    def boolean_search(self, query):
        # returns (doc_id, filename, snippet) tuples matching the boolean
        # query. no ranking here, a doc either matches or it doesnt
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

    def is_phrase_query(self, query):
        # true if the query is wrapped in double quotes, like "machine learning"
        stripped = query.strip()
        return stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2

    def phrase_search(self, query):
        # returns (doc_id, filename, snippet) tuples where the exact phrase
        # (same words, same order, right next to each other) shows up in the doc
        phrase_text = query.strip().strip('"')
        raw_words = phrase_text.split()
        phrase_terms = [self.tokenizer.stemmer.stem(w.lower()) for w in raw_words]

        matching_ids = self.index.phrase_search(phrase_terms)

        raw_query_words = {w.lower() for w in raw_words}
        results = []
        for doc_id in matching_ids:
            filename = self.processed_docs[doc_id]["filename"]
            snippet = self.get_snippet(doc_id, raw_query_words)
            results.append((doc_id, filename, snippet))

        results.sort(key=lambda r: r[0])
        return results


if __name__ == "__main__":
    engine = SearchEngine()
    engine.load_and_index("sample_corpus", cache_path="sample_corpus/.index_cache.json")
    print("Loaded from cache:", engine.loaded_from_cache)

    query = "machine learning python"
    results, corrections = engine.search(query)

    print(f"Query: {query}\n")
    for doc_id, filename, score, snippet in results:
        print(f"[{score:.4f}] doc {doc_id} - {filename}")
        print(f"...{snippet}...")