# Red Black Tree implementation by Prem
# Provides insert(score, index) and top_k(k) to match the benchmark interface
# defined in groovematch.py.

# Resources: 
# "Red Black: Insert" and related lectures on canvas
# What ".nil" is and why we use it instead of None for leaf nodes in a Red-Black Tree
    # https://cs.stackexchange.com/questions/44422/what-is-the-purpose-of-using-nil-for-representing-null-nodes
# __slots__
    # https://github.com/orgs/community/discussions/168147

RED = True
BLACK = False

class Node:
    __slots__ = ("score", "index", "color", "left", "right", "parent")

    def __init__(self, score: float, index: int, color: bool = RED):
        self.score = score
        self.index = index
        self.color = color
        self.left: "Node" = None  # type: ignore[assignment]
        self.right: "Node" = None  # type: ignore[assignment]
        self.parent: "Node" = None  # type: ignore[assignment]


class RedBlackTree:
    def __init__(self):
        self.nil = Node(0.0, -1, BLACK)
        self.nil.left = self.nil
        self.nil.right = self.nil
        self.nil.parent = self.nil
        self.root: Node = self.nil

    def insert(self, score: float, index: int) -> None:
        node = Node(score, index, RED)
        node.left = self.nil
        node.right = self.nil

        parent = self.nil
        current = self.root

        while current is not self.nil:
            parent = current
            if (score, index) < (current.score, current.index):
                current = current.left
            else:
                current = current.right

        node.parent = parent
        if parent is self.nil:
            self.root = node
        elif (score, index) < (parent.score, parent.index):
            parent.left = node
        else:
            parent.right = node

        self.rebalance(node)

    def top_k(self, k: int) -> list[tuple[float, int]]:
        result: list[tuple[float, int]] = []
        self.reverse_inorder(self.root, result, k)
        return result

    def rebalance(self, node: Node) -> None:
        while node.parent.color == RED:
            if node.parent is node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == RED:
                    node.parent.color = BLACK
                    uncle.color = BLACK
                    node.parent.parent.color = RED
                    node = node.parent.parent
                else:
                    if node is node.parent.right:
                        node = node.parent
                        self.rotate_left(node)
                    node.parent.color = BLACK
                    node.parent.parent.color = RED
                    self.rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == RED:
                    node.parent.color = BLACK
                    uncle.color = BLACK
                    node.parent.parent.color = RED
                    node = node.parent.parent
                else:
                    if node is node.parent.left:
                        node = node.parent
                        self.rotate_right(node)
                    node.parent.color = BLACK
                    node.parent.parent.color = RED
                    self.rotate_left(node.parent.parent)
        self.root.color = BLACK

    def rotate_left(self, x: Node) -> None:
        y = x.right
        x.right = y.left
        if y.left is not self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def rotate_right(self, x: Node) -> None:
        y = x.left
        x.left = y.right
        if y.right is not self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def reverse_inorder(self, node: Node, result: list[tuple[float, int]], k: int) -> None:
        if node is self.nil or len(result) >= k:
            return
        self.reverse_inorder(node.right, result, k)
        if len(result) < k:
            result.append((node.score, node.index))
        self.reverse_inorder(node.left, result, k)

if __name__ == "__main__":
    # testing basic functionality of RedBlackTree
    tree = RedBlackTree()
    tree.insert(0.95, 0)
    tree.insert(0.80, 1)
    tree.insert(0.99, 2)
    tree.insert(0.85, 3)
    tree.insert(0.92, 4)

    top_3 = tree.top_k(3)
    print("Top 3:", top_3)  #Expected output [(0.99, 2), (0.95, 0), (0.92, 4)]
