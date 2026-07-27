# a basic search engine

this is a simple search engine built from scratch, made to understand how they work under the hood.

## process
- Ingests files inside `corpus/` and tokenizes them. the tokenization process is fairly straightforward and includes
splitting, removing stopwords and lowercasing.
- a from-scratch porter stemmer takes care of the stemming process (i. e suffix removal). the results from this stemmer 
was verified by passing the same set of words to nltk's PorterStemmer.
- builds an inverted index from the tokens. note that an inverted index simply is just {word -> postings} where postings can contain anything from just the doc_id(s) where the word is present and also other payloads like the frequency.
- this inverted index is implemented using a b-tree to scale to larger corpus of data. normal dictionaries involve storing all the tokens and postings on RAM, which becomes inefficient when the data is huge.
- by default, the results are ranked using BM25. however, tf-idf/cosine similarity is also there in the code (which i might be removing soon), since it was the first algorithm i started with before switching.
- supports phrase searching by storng the indices of the words across the doc_id's in the inverted index.
- suports boolean search.
- supports fuzzy search.
- optionally, you can add --semantic while running the program from the cli and it uses semantic search. this isnt from scratch i just wanted to compare how my engine worked differently from semantic search.
- the inverted index that is built is cached so that you dont have to build it again and again.

## how to run
- go to the root of the project and run `python main.py folder_name` and ensure `folder_name` has only .txt files.
- to enable semantic search, run `python main.py folder_name --semantic` which downloads `all-MiniLM-L6-v2` from the `sentence-transformers` library so make sure you have that installed
- for phrase search, enclose your query with double quotes.

## why
- mostly just wanted to understand how search engines work, and how all the different componenents (tf-idf/bm25, inverted indexes, stemming, etc) come together.

## whats in this repo
- `core/` - the actual search engine, indexing, ranking, persistence.
- `nlp/` - tokenizer, stemmer, boolean search, fuzzy matching.
- `ingestion/` - reads files off disk into memory.
- `optional/` - embedding-based semantic search (needs sentence-transformers), and the old dict-based inverted index before i decided with btrees.
- `main.py` - well.

thats all!!