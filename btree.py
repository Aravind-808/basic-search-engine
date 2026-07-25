'''
btree.py

dictionary is good, but not as good as a btree. dicts use RAM so as long as you dont run out of it
ig you can use it BUT btrees use only disk, which is great

a btree is actually kinda slower than a dict. but the thing is, it works in a way that its actually better
to be slow than loading all that data into ram as a result of using a dict

btrees have a root, and consequent levels to them. in those levels, there are nodes. these nodes
must contain anywhere between t-1 to 2t-1 keys

when yu go down a level, you utilize a "disk read" tp bring the node into ram. so the entire point of designing
a btree is to minimize these levels so that even with millions of data (keys), you perform only a few disk
reads.
'''

class BTreeNode:
    def __init__(self, leaf = True):
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.children = []

    def search(self, key):
        return self.search_btree(self.root, key)

    def search_btree(self, node, key):
        # what we are gonna do is essentially find the key index "i" first
        # because of how b trees are structured, if the key exists, it exists at
        # position i at all levels

        # after that its just a matter of searching all levels
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i+=1

        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]

        if node.leaf:
            return None

        return self.search_btree(node.children[i], key)

    