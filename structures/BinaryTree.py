import numpy as np

from BTNode import *
from DS import *


class BinaryTree(DS):
    def __init__(self, elements, frequencies=None, pessimistic=True):
        DS.__init__(self)
        self.elements = elements
        self.frequencies = frequencies
        self.pessimistic = pessimistic
        self.make_tree(elements, frequencies, pessimistic)

    def make_tree(self, elements, frequencies, pessimistic):
        if pessimistic:
            self.root, self.height = self.generate_balance_tree(elements)
        else:

            self.root, self.height = self.construct_optimal_bst_n2(elements, frequencies)

    def insert(self, key, frequency):

        if key not in self.elements:
            log = "{0} inserted. N = {1}".format(key, len(self.elements))
            self.elements.append(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.append(frequency)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.make_tree(self.elements, self.frequencies, self.pessimistic)

        else:
            log = "{0} insertion failed. N = {1}".format(key, len(self.elements))
        # with open("RandomOrderDynamic/log.txt", 'a') as writer:
        #     writer.write(log+ "\n")


    def delete(self, key):
        if key in self.elements:
            log = "{0} deleted. N = {1}".format(key, len(self.elements))
            idx = self.elements.index(key)
            self.elements.remove(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.pop(idx)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.make_tree(self.elements, self.frequencies, self.pessimistic)
        else:
            log = "{0} deletion failed. N = {1}".format(key, len(self.elements))
        # with open("RandomOrderDynamic/log.txt", 'a') as writer:
        #     writer.write(log + "\n")



    def generate_balance_tree(self, elements):
        ordered_elements = sorted(elements)

        if len(ordered_elements) == 0:
            return None, 0

        if len(ordered_elements) == 1:
            root = BTNode(ordered_elements[0])
            return root, 1
        else:
            root_index = int(len(ordered_elements) / 2)
            left_set = ordered_elements[:root_index]
            right_set = ordered_elements[root_index + 1:]

            root = BTNode(ordered_elements[root_index])
            left_sub_tree, left_height = self.generate_balance_tree(left_set)
            right_sub_tree, right_height = self.generate_balance_tree(right_set)

            root.set_left_child(left_sub_tree)
            root.set_right_child(right_sub_tree)

            if left_sub_tree is not None:
                left_sub_tree.set_parent(root)
            if right_sub_tree is not None:
                right_sub_tree.set_parent(root)

            height = max(left_height, right_height) + 1

            return root, height

    def construct_optimal_bst(self, ordered_elements, freq):
        n = len(ordered_elements)

        # Create a 2D table to store the cost of optimal BSTs
        cost = [[0 for j in range(n)] for i in range(n)]

        # Create a 2D table to store the root of optimal BSTs
        root = [[None for j in range(n)] for i in range(n)]

        # Initialize the diagonal of the cost table with the frequency of the keys
        for i in range(n):
            cost[i][i] = freq[i]
            root[i][i] = i

        # Build the cost and root tables for all subtrees of size 2 to n
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = i + L - 1
                cost[i][j] = float('inf')
                # Try all possible roots in the subtree
                for k in range(i, j + 1):
                    # Compute the cost of the subtree with root k
                    # c = (0 if k == i else cost[i][k - 1]) + \
                    #     sum(freq[k:j + 1]) + \
                    #     (0 if k == j else cost[k + 1][j])
                    left_cost = (0 if k == i else cost[i][k - 1])
                    right_cost = (0 if k == j else cost[k + 1][j])
                    c = left_cost + right_cost + sum(freq[i:j + 1])
                    # Update the minimum cost and root of the subtree
                    if c < cost[i][j]:
                        cost[i][j] = c
                        root[i][j] = k

        return self.get_optimal_tree_from_roots(ordered_elements, root, 0, n - 1)

    def get_optimal_tree_from_roots(self, ordered_elements, roots, i, j):
        if j < i:
            return None, 0
        if i == j:
            return BTNode(ordered_elements[i]), 1
        root_index = roots[i][j]
        root = BTNode(ordered_elements[root_index])
        left_tree, left_height = self.get_optimal_tree_from_roots(ordered_elements, roots, i, root_index - 1)
        right_tree, right_height = self.get_optimal_tree_from_roots(ordered_elements, roots, root_index + 1, j)

        root.set_left_child(left_tree)
        root.set_right_child(right_tree)

        if left_tree is not None:
            left_tree.set_parent(root)
        if right_tree is not None:
            right_tree.set_parent(root)

        height = max(left_height, right_height) + 1

        return root, height

    def search(self, key_Value, __splay_cost__=False):
        node = self.root
        cost = 1
        while True:
            if node is None:
                print("Couldn't find key {0}".format(key_Value))
                sorted_keys = sorted(self.elements.copy())
                print("List of live keys: ", sorted_keys)
                return None

            if node.value == key_Value:
                break
            if key_Value < node.value:
                node = node.left
                cost += 1
            else:
                node = node.right
                cost += 1



        return node, cost

    def BFS(self):
        levels = []
        expanding_nodes = [self.root]
        levels.append([self.root])
        while len(expanding_nodes) > 0:
            next_level = []
            for node in expanding_nodes:
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            if len(next_level) > 0:
                levels.append(next_level)
            expanding_nodes = next_level
        return levels

    def construct_optimal_bst_n(self, ordered_elements, freq):
        n = len(ordered_elements)

        # Create a 2D table to store the cost of optimal BSTs
        cost = [[0 for j in range(n + 1)] for i in range(n + 1)]

        # Create a 2D table to store the root of optimal BSTs
        root = [[None for j in range(n + 1)] for i in range(n + 1)]

        # Initialize the diagonal of the cost table with the frequency of the keys
        for i in range(n):
            cost[i][i] = freq[i]
            root[i][i] = i

        fsum = [[0 for _ in range(n)] for __ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    fsum[i][j] = freq[i]
                else:
                    fsum[i][j] = fsum[i][j - 1] + freq[j]

        minCost, cost, root = self.optCost_memoized(freq, 0, n - 1, cost, root, fsum)

        return self.get_optimal_tree_from_roots(ordered_elements, root, 0, n - 1)

    def construct_optimal_bst_n2(self, elements, frequencies):
        element_frequencies = [[elements[i], frequencies[i]] for i in range(len(elements))]
        element_frequencies = sorted(element_frequencies, key=lambda t: t[0])
        ordered_elements = []
        freq = []
        for v in element_frequencies:
            ordered_elements.append(v[0])
            freq.append(v[1])
        n = len(ordered_elements)

        # Create a 2D table to store the cost of optimal BSTs
        cost = [[0 for j in range(n + 1)] for i in range(n + 1)]

        # Create a 2D table to store the root of optimal BSTs
        root = [[None for j in range(n + 1)] for i in range(n + 1)]

        # Initialize the diagonal of the cost table with the frequency of the keys
        for i in range(n):
            cost[i][i] = freq[i]
            root[i][i] = i

        fsum = [[0 for _ in range(n)] for __ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    fsum[i][j] = freq[i]
                else:
                    fsum[i][j] = fsum[i][j - 1] + freq[j]

        minCost, cost, root = self.calc_optCost(n, freq, cost, root, fsum)

        return self.get_optimal_tree_from_roots(ordered_elements, root, 0, n - 1)

    def calc_optCost(self, n, freq, cost, roots, freq_sum):
        i = n - 1
        while i >= 0:
            for j in range(i, n):
                c, cost, roots = self.optCost_memoized(freq, i, j, cost, roots, freq_sum)
            i -= 1
        return cost[0][n - 1], cost, roots

    def optCost_memoized(self, freq, i, j, cost, roots, freq_sum):
        if i > j:
            cost[i][j] = 0
            return cost[i][j], cost, roots
        if cost[i][j]:
            return cost[i][j], cost, roots

        # Initialize minimum value
        Min = float('inf')

        for r in range(i, j + 1):
            c1, cost, roots = self.optCost_memoized(freq, i, r - 1, cost, roots, freq_sum)
            c2, cost, roots = self.optCost_memoized(freq, r + 1, j, cost, roots, freq_sum)
            c = c1 + c2 + freq_sum[i][j]
            if c < Min:
                Min = c

                cost[i][j] = c
                roots[i][j] = r

        # Return minimum value
        return cost[i][j], cost, roots
