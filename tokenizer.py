'''
tokenizer.py

takes the output of DocumentCollector.file_data (list of dicts with
id, filename, content, path) and produces tokenized documents ready
for term-frequency / TF-IDF computation.
'''

from stemming import PorterStemmer

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "in", "on", "of", "to",
    "it", "this", "that", "for", "with", "as", "was", "were", "are"
}


class Tokenizer:
    def __init__(self, remove_stopwords=True, use_stemming=True, min_length=1):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.min_length = min_length
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        text = text.lower()
        tokens = []
        current = ""
        for char in text:
            if char.isalnum():
                current += char
            else:
                if current:
                    tokens.append(current)
                    current = ""
        if current:
            tokens.append(current)

        if self.min_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_length]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]

        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

    def process_documents(self, file_data):
        """
        takes DocumentCollector.file_data and returns:
        {
            doc_id: {
                "filename": ...,
                "tokens": [...],
                "term_freq": {...}
            }
        }
        """
        processed = {}
        for doc in file_data:
            tokens = self.tokenize(doc["content"])
            processed[doc["id"]] = {
                "filename": doc["filename"],
                "tokens": tokens,
                "term_freq": self.term_frequencies(tokens)
            }
        return processed

    def term_frequencies(self, tokens):
        freqs = {}
        for token in tokens:
            freqs[token] = freqs.get(token, 0) + 1
        return freqs


if __name__ == "__main__":
    from file_loader import DocumentCollector

    collector = DocumentCollector()
    collector.load_files(r"C:\Users\admin\agent_folder")

    tokenizer = Tokenizer()
    processed = tokenizer.process_documents(collector.file_data)

    for doc_id, data in processed.items():
        print(doc_id, data["filename"], data["tokens"][:10])