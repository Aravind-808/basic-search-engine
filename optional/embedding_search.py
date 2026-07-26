'''
embedding_search.py

uhhhhhhhhhhhhhhhhh not really true to the "from scratch" faith. just a little something 
for me use to see how semantic search will work in this setting 

Semantic search using sentence-transformers. Encodes each document (and
each query) into a dense vector such that documents with similar MEANING
end up close together in vector space, even with zero shared words.

needs pip install sentence-transformers
'''

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.doc_ids = [] # doc_ids in the same order as doc_embeddings
        self.doc_embeddings = None  # numpy array, shape (num_docs, embedding_dim)

    def build(self, raw_content):
        """
        raw_content: {doc_id: original_text} - the UNTOKENIZED text,
        since sentence-transformers models expect natural language,
        not stemmed/stopword-filtered tokens.
        """
        self.doc_ids = list(raw_content.keys())
        texts = [raw_content[doc_id] for doc_id in self.doc_ids]

        # normalize_embeddings=True makes vectors unit length, so cosine
        # similarity reduces to a plain dot product below
        self.doc_embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def search(self, query, top_k=5):
        """
        Returns a ranked list of (doc_id, score) tuples, highest similarity
        first. Score is cosine similarity, ranges roughly -1 to 1.
        """
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]

        # since every vector is already unit length, cosine similarity
        # between query and every doc is just one matrix-vector dot product
        similarities = self.doc_embeddings @ query_embedding

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[i], float(similarities[i])) for i in ranked_indices]


if __name__ == "__main__":
    from ingestion.file_loader import DocumentCollector

    collector = DocumentCollector()
    collector.load_files("topic_corpus")
    raw_content = {doc["id"]: doc["content"] for doc in collector.file_data}
    filenames = {doc["id"]: doc["filename"] for doc in collector.file_data}

    engine = EmbeddingSearch()
    engine.build(raw_content)

    query = "how do neural networks understand images"
    results = engine.search(query)

    print(f"Query: {query}\n")
    for doc_id, score in results:
        print(f"  [{score:.4f}] {filenames[doc_id]}")