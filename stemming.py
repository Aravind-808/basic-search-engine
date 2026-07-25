"""
stemming.py
 
A from-scratch implementation of the Porter Stemming Algorithm (Porter, 1980).
No external libraries used - pure Python.

forgot to add this: before looking at the code, PLEASE LOOK AT THE
COMICALLY BIG DESCRIPTION of the porter stemming algo.

step 2 to step 4 are just massive massive lookup tables and rules applied to them 
based on their measure vakue
"""
 
 
class PorterStemmer:
 
    STEP2_SUFFIXES = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ]
 
    STEP3_SUFFIXES = [
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ]
 
    STEP4_SUFFIXES = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent",
        "ion",  # special case: needs stem ending in s/t, handled separately
        "ou", "ism", "ate", "iti", "ous", "ive", "ize"
    ]
 
    def is_consonant(self, word, i):
        char = word[i]
        if char in "aeiou":
            return False
        if char == "y":
            if i == 0:
                return True
            return not self.is_consonant(word, i - 1)
        return True
 
    def measure(self, word):
        """Counts the number of VC sequences (the 'm' value)."""
        if not word:
            return 0
        form = "".join(
            "C" if self.is_consonant(word, i) else "V"
            for i in range(len(word))
        )
        m = 0
        i = 0
        while i < len(form) and form[i] == "C":
            i += 1
        while i < len(form):
            while i < len(form) and form[i] == "V":
                i += 1
            if i >= len(form):
                break
            m += 1
            while i < len(form) and form[i] == "C":
                i += 1
        return m
 
    def contains_vowel(self, word):
        return any(not self.is_consonant(word, i) for i in range(len(word)))
 
    def ends_with_double_consonant(self, word):
        if len(word) < 2:
            return False
        return word[-1] == word[-2] and self.is_consonant(word, len(word) - 1)
 
    def cvc(self, word):
        """True if the last 3 letters are consonant-vowel-consonant,
        and the final consonant is not w, x, or y."""
        if len(word) < 3:
            return False
        i = len(word) - 1
        if not self.is_consonant(word, i):
            return False
        if word[i] in "wxy":
            return False
        if self.is_consonant(word, i - 1):
            return False
        if not self.is_consonant(word, i - 2):
            return False
        return True
 
    # Step 1a: plurals
 
    def step1a(self, word):
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word
 
    # Step 1b: ed / ing 
 
    def step1b(self, word):
        if word.endswith("eed"):
            stem = word[:-3]
            if self.measure(stem) > 0:
                return stem + "ee"
            return word
 
        for suffix in ("ed", "ing"):
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.contains_vowel(stem):
                    if stem.endswith(("at", "bl", "iz")):
                        return stem + "e"
                    if self.ends_with_double_consonant(stem) and stem[-1] not in "lsz":
                        return stem[:-1]
                    if self.measure(stem) == 1 and self.cvc(stem):
                        return stem + "e"
                    return stem
                return word
        return word
 
    # Step 1c: y -> i
 
    def step1c(self, word):
        if word.endswith("y"):
            stem = word[:-1]
            if self.contains_vowel(stem):
                return stem + "i"
        return word
 
    #Step 2: derivational suffixes (m > 0)
 
    def step2(self, word):
        for suffix, replacement in self.STEP2_SUFFIXES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.measure(stem) > 0:
                    return stem + replacement
                return word
        return word
 
    # Step 3: more derivational suffixes (m > 0)
 
    def step3(self, word):
        for suffix, replacement in self.STEP3_SUFFIXES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.measure(stem) > 0:
                    return stem + replacement
                return word
        return word
 
    # Step 4: suffix removal (m > 1)
 
    def step4(self, word):
        for suffix in self.STEP4_SUFFIXES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == "ion":
                    if self.measure(stem) > 1 and stem.endswith(("s", "t")):
                        return stem
                    continue
                if self.measure(stem) > 1:
                    return stem
                return word
        return word
 
    # Step 5a: remove trailing e
 
    def step5a(self, word):
        if word.endswith("e"):
            stem = word[:-1]
            m = self.measure(stem)
            if m > 1:
                return stem
            if m == 1 and not self.cvc(stem):
                return stem
        return word
 
    # Step 5b: remove doubled l
 
    def step5b(self, word):
        if self.measure(word) > 1 and word.endswith("ll"):
            return word[:-1]
        return word
 
    def stem(self, word):
        word = word.lower()
        if len(word) <= 2:
            return word
 
        word = self.step1a(word)
        word = self.step1b(word)
        word = self.step1c(word)
        word = self.step2(word)
        word = self.step3(word)
        word = self.step4(word)
        word = self.step5a(word)
        word = self.step5b(word)
        return word
 


if __name__ == '__main__':    
    # to test
    import nltk
    from nltk.stem import PorterStemmer
    porterstemmer = nltk.stem.PorterStemmer()
    stemmer = PorterStemmer()
 
    test_words = [
        "caresses", "ponies", "caress", "cats",
        "agreed", "feed", "plastered", "hopping", "filing",
        "happy", "sky",
        "relational", "conditional", "hopefulness",
        "triplicate", "formative", "hopeful",
        "revival", "adoption", "allowance",
        "probate", "cease", "cave", "controll"
    ]
 
    for w in test_words:
        if porterstemmer.stem(w) != stemmer.stem(w):
            print("Test Failed, error in your stemmer")
            exit(0)

    print("All testcases passed!!")
