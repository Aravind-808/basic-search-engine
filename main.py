'''
main.py

Simple command-line interface for the search engine.

Usage:
    python main.py <folder_path>

Then type queries at the prompt. Type 'quit' or 'exit' to stop.
'''

import sys
from search_engine import SearchEngine


def print_results(results):
    if not results:
        print("  No matches found.\n")
        return
    for doc_id, filename, score in results:
        print(f"  [{score:.4f}] doc {doc_id} - {filename}")
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
    while True:
        query = input("> ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        results = engine.search(query)
        print_results(results)


if __name__ == "__main__":
    main()