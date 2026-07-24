# boolean search for my search engine

'''
supports queries like 
    - x AND y
    - x OR y
    - x OR y AND z
'''

ops = {"AND", "NOT", "OR"}

def parse(query, tokenizer):
    raw_tokens = query.split()
    parsed_stuff = []
    last_op = None 

    for tok in raw_tokens:
        upper = tok.upper()
        if upper in ops:
            last_op = upper
            continue

        stemmed = tokenizer.stemmer.stem(tok.lower())

        op = last_op if last_op else "AND"
        parsed_stuff.append((op, stemmed))
        last_op = None

    return parsed_stuff

def boolean_search(query, tokenizer, index, all_doc_ids):
    parsed = parse(query, tokenizer)
    if not parsed:
        return set()

    # the first term establishes the base set (its own "operator" only
    # matters if it's NOT, meaning "all docs except this term")
    first_op, first_term = parsed[0]
    first_set = set(index.get_postings(first_term).keys())
    result = (all_doc_ids - first_set) if first_op == "NOT" else first_set

    for op, term in parsed[1:]:
        term_set = set(index.get_postings(term).keys())
        if op == "AND":
            result = result & term_set
        elif op == "OR":
            result = result | term_set
        elif op == "NOT":
            result = result - term_set
 
    return result

if __name__ == "__main__":
    from file_loader import DocumentCollector
    from tokenizer import Tokenizer
    from inverted_index import InvertedIndex
 
    collector = DocumentCollector()
    collector.load_files("test_docs")
 
    tokenizer = Tokenizer()
    processed = tokenizer.process_documents(collector.file_data)
 
    index = InvertedIndex()
    index.build(processed)
 
    all_ids = set(processed.keys())
 
    test_queries = [
        "python AND learning",
        "python OR cooking",
        "machine NOT deep",
        "python NOT snake",
    ]
 
    for q in test_queries:
        matches = boolean_search(q, tokenizer, index, all_ids)
        filenames = [processed[d]["filename"] for d in matches]
        print(f"{q}\t\t{filenames}")

