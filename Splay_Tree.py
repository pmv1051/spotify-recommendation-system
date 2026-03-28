# Splay Tree implementation by Jenish
# Includes top_k and insert integrations into the benchmark 

# Resources used:
# 
#



class Node:
    def __init__(self, score, index):

        self.score = score
        self.index = index

        self.right = None
        self.left = None
        self.parent = None


class SplayTree:

    # constructor skeleton
    def __init__(self):
    
        self.root = None

    def insert(self, score, index):
        #insert logic

        node = Node(score, index)

        if self.root is None:
            self.root = node
            return
        
        #traverse down BST insert
        c = self.root

        start = True

        while start:
            if (score, index) < (c.score, c.index):
                if c.left is None:
                    c.left = node
                    node.parent  = c

                    start = False
                    break
                c = c.left
            
            else:
                if c.right is None:
                    c.right = node
                    node.parent = c

                    start = False
                    break

                c = c.right
        
        self.splay(node)


    def left_rotate(self, x: Node):
        #left rotation logic
        if x is None or x.right is None:
            return
        
        y = x.parent
        right = x.right

        x.right = right.left
        if right.left is not None:
            right.left.parent = x

        right.parent = y
        
        if y is None: 
            self.root = right

        else:
            if y.left == x:
                y.left = right

            else:
                y.right = right

        right.left = x
        x.parent = right
            

    def right_rotate(self, x):
        #right rotation logic
        if x is None or x.left is None:
            return 
        
        y = x.parent
        left = x.left

        x.left = left.right

        if left.right is not None:
            left.right.parent = x

        left.parent = y
        if y is None: 
            self.root = left

        else: 
            if y.right == x:
                y.right = left

            else: 
                y.left = left
        
        left.right = x
        x.parent = left

        

    def top_k(self, k):
        # iterative reverse in-order (right -> node -> left)
        # avoids RecursionError on trees
        result = []
        stack = []
        node = self.root

        while (stack or node is not None) and len(result) < k:
            while node is not None:
                stack.append(node)
                node = node.right

            node = stack.pop()
            result.append((node.score, node.index))
            node = node.left

        return result
    

    def splay(self, x):
 
        while x.parent is not None:

            #zig 
            if x.parent.parent is None:
                if x == x.parent.left:
                    self.right_rotate(x.parent)
                else: 
                    self.left_rotate(x.parent)

            #zigzig
            elif x == x.parent.left and x.parent == x.parent.parent.left:
                self.right_rotate(x.parent.parent)
                self.right_rotate(x.parent)

            elif x == x.parent.right and x.parent == x.parent.parent.right:
                self.left_rotate(x.parent.parent)
                self.left_rotate(x.parent)
            
            #zigzag
            elif x == x.parent.right and x.parent == x.parent.parent.left:
                self.left_rotate(x.parent)
                self.right_rotate(x.parent)

            elif x == x.parent.left and x.parent == x.parent.parent.right:
                self.right_rotate(x.parent)
                self.left_rotate(x.parent)
