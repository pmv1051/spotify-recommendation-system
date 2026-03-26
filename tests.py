import unittest
from RB_Tree import RedBlackTree


class TestRedBlackTree(unittest.TestCase):

    def test_empty_tree(self):
        tree = RedBlackTree()
        self.assertEqual(tree.top_k(5), [])

    def test_insert_single(self):
        tree = RedBlackTree()
        tree.insert(0.5, 0)
        self.assertEqual(tree.top_k(1), [(0.5, 0)])

    def test_top_k(self):
        tree = RedBlackTree()
        tree.insert(0.95, 0)
        tree.insert(0.80, 1)
        tree.insert(0.99, 2)
        tree.insert(0.85, 3)
        tree.insert(0.92, 4)
        self.assertEqual(tree.top_k(3), [(0.99, 2), (0.95, 0), (0.92, 4)])

    def test_top_k_larger_than_size(self):
        tree = RedBlackTree()
        tree.insert(0.1, 0)
        tree.insert(0.2, 1)
        self.assertEqual(tree.top_k(10), [(0.2, 1), (0.1, 0)])

    def test_duplicate_scores(self):
        tree = RedBlackTree()
        tree.insert(0.5, 0)
        tree.insert(0.5, 1)
        result = tree.top_k(2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
