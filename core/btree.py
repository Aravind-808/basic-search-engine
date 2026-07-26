'''
btree.py

dictionary is good, but not as good as a btree. dicts use RAM so as long as you dont run out of it
ig you can use it BUT btrees use only disk, which is great

a btree is actually kinda slower than a dict. but the thing is, it works in a way that its actually better
to be slow than loading all that data into ram as a result of using a dict

btrees have a root, and consequent levels to them. in those levels, there are nodes. these nodes
must contain anywhere between t-1 to 2t-1 keys

when you go down a level, you utilize a "disk read" tp bring the node into ram. so the entire point of designing
a btree is to minimize these levels so that even with millions of data (keys), you perform only a few disk
reads.
'''

class BTreeNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []      # sorted list of terms
        self.values = []    # values[i] corresponds to keys[i]
        self.children = []  # only non-empty if not a leaf, len == len(keys)+1


class BTree:
    def __init__(self, order=3):
        self.order = order  # t, the minimum degree
        self.root = BTreeNode(leaf=True)

    def search(self, key):
        return self.search_btree(self.root, key)

    def search_btree(self, node, key):
        # what we are gonna do is essentially find the key index "i" first
        # because of how b trees are structured, if the key exists, it exists at
        # position i at all levels

        # after that its just a matter of searching all levels
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]

        if node.leaf:
            return None

        return self.search_btree(node.children[i], key)

    def insert(self, key, value):
        root = self.root

        # if the root is already full (2t-1 keys), it has to split BEFORE we
        # even start descending - this is what keeps the tree growing upward
        # (adding a level) instead of some branches getting deeper than others
        if len(root.keys) == 2 * self.order - 1:
            new_root = BTreeNode(leaf=False)
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.root = new_root

        self.insert_non_full(self.root, key, value)

    def split_child(self, parent, child_index):
        # takes a full child of parent, splits it into two half-full nodes,
        # and pushes the middle key/value up into parent
        t = self.order
        full_child = parent.children[child_index]
        new_child = BTreeNode(leaf=full_child.leaf)

        mid_key = full_child.keys[t - 1]
        mid_value = full_child.values[t - 1]

        new_child.keys = full_child.keys[t:]
        new_child.values = full_child.values[t:]
        full_child.keys = full_child.keys[:t - 1]
        full_child.values = full_child.values[:t - 1]

        if not full_child.leaf:
            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

        parent.children.insert(child_index + 1, new_child)
        parent.keys.insert(child_index, mid_key)
        parent.values.insert(child_index, mid_value)

    def insert_non_full(self, node, key, value):
        if node.leaf:
            # find where key belongs, or where it already exists so we can overwrite
            pos = 0
            while pos < len(node.keys) and node.keys[pos] < key:
                pos += 1
            if pos < len(node.keys) and node.keys[pos] == key:
                node.values[pos] = value
                return
            node.keys.insert(pos, key)
            node.values.insert(pos, value)
            return

        i = len(node.keys) - 1
        while i >= 0 and key < node.keys[i]:
            i -= 1

        if i >= 0 and node.keys[i] == key:
            node.values[i] = value
            return

        i += 1  # child index to descend into

        if len(node.children[i].keys) == 2 * self.order - 1:
            self.split_child(node, i)
            if key > node.keys[i]:
                i += 1

        self.insert_non_full(node.children[i], key, value)

    # traversal 
    def inorder(self):
        result = []
        self.inorder_node(self.root, result)
        return result

    def inorder_node(self, node, result):
        for i in range(len(node.keys)):
            if not node.leaf:
                self.inorder_node(node.children[i], result)
            result.append((node.keys[i], node.values[i]))
        if not node.leaf:
            self.inorder_node(node.children[-1], result)

    def height(self):
        h = 1
        node = self.root
        while not node.leaf:
            h += 1
            node = node.children[0]
        return h


if __name__ == "__main__":
    import random

    tree = BTree(order=2)  # small order so splits happen often, easy to eyeball

    words = ["python", "search", "engine", "index", "query", "rank",
              "token", "stem", "vector", "cosine", "boolean", "phrase"]

    for w in words:
        tree.insert(w, f"postings-for-{w}")

    print("Height:", tree.height())
    print()
    print("Inorder (should be alphabetically sorted):")
    for key, value in tree.inorder():
        print(f"  {key:10} -> {value}")

    print()
    print("search('token'):", tree.search("token"))
    print("search('missing'):", tree.search("missing"))

    print()
    print("Correctness check against dict on 500 random keys")
    reference = {}
    big_tree = BTree(order=3)
    keys = [f"term{i}" for i in range(500)]
    random.shuffle(keys)
    for k in keys:
        big_tree.insert(k, k.upper())
        reference[k] = k.upper()

    mismatches = 0
    for k in reference:
        if big_tree.search(k) != reference[k]:
            mismatches += 1
    print(f"Mismatches: {mismatches} (should be 0)")
    print(f"Inorder length: {len(big_tree.inorder())} (should be 500)")
    sorted_keys = [k for k, v in big_tree.inorder()]
    print("Inorder actually sorted:", sorted_keys == sorted(reference.keys()))