# Splay Tree implementation by Jenish

class Node:
    def __init__(self, key):

        self.key = key
        self.left = None
        Self.right = None

class SplayTree:

    #insert constructor

    def splay(root, key):
        if root is None :
            return new_node(key)
        
        if root.key == key:

            return root
        
        if root.key > key:
            if root.left is None:
                return root
            
            #continue implementation
