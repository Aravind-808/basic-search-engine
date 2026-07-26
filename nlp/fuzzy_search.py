'''
fuzzy_search.py

suggest a closest known term when the typed term in the query isnt in
the inverted index at all

edit distance flashbacks
'''


def levenshtein_distance(a, b):
    # dp[i][j] = edit distance between a[:i] and b[:j]
    m, n = len(a), len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i  
    for j in range(n + 1):
        dp[0][j] = j  

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       
                dp[i][j - 1] + 1,       
                dp[i - 1][j - 1] + cost  
            )

    return dp[m][n]


def find_closest_terms(term, vocabulary, max_distance=2, max_results=3):
    # returns up to max_results vocabulary terms within max_distance
    # of the given term, sorted by closeness (smallest distance first).

    candidates = []
    for vocab_term in vocabulary:
        dist = levenshtein_distance(term, vocab_term)
        if dist <= max_distance:
            candidates.append((vocab_term, dist))

    candidates.sort(key=lambda pair: pair[1])
    return [term for term, dist in candidates[:max_results]]


if __name__ == "__main__":
    # sanity checks 
    tests = [
        ("kitten", "sitting", 3),
        ("python", "pythno", 2),
        ("flaw", "lawn", 2),
        ("same", "same", 0),
    ]
    for a, b, expected in tests:
        got = levenshtein_distance(a, b)
        status = "ok" if got == expected else "no"
        print(f"{status}: distance({a}, {b}) = {got} (expected {expected})")

    vocab = ["python", "search", "engine", "machine", "learning", "learn", "cooking"]
    print()
    print("closest to 'pythno':", find_closest_terms("pythno", vocab))
    print("closest to 'serach':", find_closest_terms("serach", vocab))
    print("closest to 'enging':", find_closest_terms("enging", vocab))