import sys
from search_engine import SearchEngine


def print_ranked_results(results, corrections):
    if corrections:
        for original, corrected in corrections:
            print(f"(no match for '{original}', using closest term '{corrected}')")

    if not results:
        print("No matches found.\n")
        return

    for doc_id, filename, score, snippet in results:
        print(f"  [{score:.4f}] doc {doc_id} - {filename}")
        print(f"...{snippet}...")
    print()


def print_boolean_results(results):
    if not results:
        print("No matches found.\n")
        return

    for doc_id, filename, snippet in results:
        print(f"doc {doc_id} - {filename}")
        print(f"...{snippet}...")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    print(f"Indexing documents in '{folder_path}'...")
    engine = SearchEngine()
    engine.load_and_index(folder_path)

    doc_count = len(engine.processed_docs)
    term_count = len(engine.index.vocabulary())
    print(f"Indexed {doc_count} documents, {term_count} unique terms.\n")

    print("Type a search query (or 'quit' to exit).")
    print("Use AND / OR / NOT for boolean search, e.g. 'python AND learning'.")
    while True:
        query = input("> ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        if engine.is_boolean_query(query):
            results = engine.boolean_search(query)
            print_boolean_results(results)
        else:
            results, corrections = engine.search(query)
            print_ranked_results(results, corrections)


if __name__ == "__main__":
    main()