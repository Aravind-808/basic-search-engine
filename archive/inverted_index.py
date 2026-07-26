'''
inverted_index.py

Builds an inverted index from tokenized documents (output of
Tokenizer.process_documents), and computes IDF for TF-IDF scoring.
Also builds a positional index to support exact phrase search.
'''

import math


class InvertedIndex:
    def __init__(self):
        self.index = {}       # term -> {doc_id: frequency}
        self.positions = {}   # term -> {doc_id: [positions in token stream]}
        self.doc_count = 0    # total number of documents (N)

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

            # positions come from the ordered token list itself - the
            # index in the (already stopword-filtered, stemmed) list
            for position, term in enumerate(data["tokens"]):
                self.positions.setdefault(term, {}).setdefault(doc_id, []).append(position)

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

    def get_positions(self, term, doc_id):
        """Returns a list of positions where term occurs in doc_id, or []."""
        return self.positions.get(term, {}).get(doc_id, [])

    def phrase_search(self, phrase_terms):
        """
        phrase_terms: list of already-stemmed terms, in order, e.g. ["machin", "learn"]
        Returns the set of doc_ids where these terms appear consecutively,
        in this exact order, in the token stream.
        """
        if not phrase_terms:
            return set()

        # start from candidate docs that contain the FIRST term at all
        first_term = phrase_terms[0]
        candidate_docs = set(self.positions.get(first_term, {}).keys())

        # narrow down to docs containing every term in the phrase
        for term in phrase_terms[1:]:
            candidate_docs &= set(self.positions.get(term, {}).keys())

        matching_docs = set()
        for doc_id in candidate_docs:
            first_positions = self.positions[first_term][doc_id]
            for start_pos in first_positions:
                if self.sequence_matches(phrase_terms, doc_id, start_pos):
                    matching_docs.add(doc_id)
                    break  # no need to check other start positions in this doc

        return matching_docs

    def sequence_matches(self, phrase_terms, doc_id, start_pos):
        """
        checks whether phrase_terms[1:] appear at consecutive positions
        right after start_pos in doc_id (start_pos is where phrase_terms[0] is).
        """
        for offset, term in enumerate(phrase_terms[1:], start=1):
            expected_pos = start_pos + offset
            term_positions = self.positions.get(term, {}).get(doc_id, [])
            if expected_pos not in term_positions:
                return False
        return True


if __name__ == "__main__":
    from file_loader import DocumentCollector
    from tokenizer import Tokenizer

    collector = DocumentCollector()
    collector.load_files("corpus")

    tokenizer = Tokenizer()
    processed = tokenizer.process_documents(collector.file_data)

    index = InvertedIndex()
    index.build(processed)

    print("Vocabulary size:", len(index.vocabulary()))
    for term in ["diabetes", "fatigue", "treatment"]:
        print(f"{term:10} postings={index.get_postings(term)}  df={index.document_frequency(term)}  idf={index.idf(term):.4f}")