import numpy as np
from structures.BTNode import *
from structures.DS import *


class BinaryTree(DS):
    


    """
    Initialize the BinaryTree wrapper over BTNode with either a balanced 
    tree or an optimal BST, depending on the 'pessimistic' flag.

    Paramters:
    - elements : list of keys
    - frequencies : list of access frequencies aligned with 'elements'
    - pessimistic : bool flag
        - True -> build a balanced BST (frequency-agnostic)
        - False -> build an optimal BST using frequencies

    Side effects:
    - Sets self.root and self.height via make_tree().
    """
    def __init__(self, elements, frequencies=None, pessimistic=True):
        DS.__init__(self)
        self.elements = elements
        self.frequencies = frequencies
        self.pessimistic = pessimistic
        self.size = len(elements) if elements else 0
        self.make_tree(elements, frequencies, pessimistic)



    """
    Build the tree under the current config.

    Parameters: same as __init__
    
    Sets: 
    - self.root : BTNode or None
    - self.height : int (0 if empty)
    
    Notes:
    - Balanced tree ignores 'frequencies'
    - Optimal BST requires 'frequences' (same length as 'elements')
    """
    def make_tree(self, elements, frequencies, pessimistic):
        if pessimistic:
            self.root, self.height = self.generate_balance_tree(elements)
        else:

            self.root, self.height = self.construct_optimal_bst_n2(elements, frequencies)



    """
    Insert a key with an associated frequency, then rebuild the tree.

    Parameters:
    - key : orderable key to insert (must not already exist)
    - frequency : weight for the key

    Behavior:
        - If key is new:
            - Appends key and frequency.
            - Normalizes 'self.frequencies' to sum to 1.
            - Rebuilds the tree using current 'pessimistic' mode.
        - If key exists:
            - No-op (logs failure message if logging is enabled).

    Side Effects:
    - Mutates self.elements and self.frequencies.
    - Rebuilds the entire tree (can be O(n log n) for balanced or higher for OBST).
    """
    def insert(self, key, frequency):

        if key not in self.elements:
            log = "{0} inserted. N = {1}".format(key, len(self.elements))
            self.elements.append(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.append(frequency)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.size += 1
            self.make_tree(self.elements, self.frequencies, self.pessimistic)

        else:
            log = "{0} insertion failed. N = {1}".format(key, len(self.elements))
        # with open("RandomOrderDynamic/log.txt", 'a') as writer:
        #     writer.write(log+ "\n")


    """
    Returns the current number of nodes in the tree.

    """
    def get_size(self):
        return self.size

    """
    Delete a key (if present), then rebuild the tree. 

    Parameters:
    - key : orderable key to remove

    Behavior:
    - If key exists:
        - Removes key and its frequency.
        - Renormalizes remaining frequencies to sum to 1.
        - Rebuilds the tree using current 'pessimistic' mode.
    - If key does not exist:
        - No-op (logs failure message if logging is enabled).

    Side Effects:
    - Mutates self.elements and self.frequencies.
    - Rebuilds the entire tree (cost depends on mode).
    """
    def delete(self, key):
        if key in self.elements:
            log = "{0} deleted. N = {1}".format(key, len(self.elements))
            idx = self.elements.index(key)
            self.elements.remove(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.pop(idx)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.size -= 1
            self.make_tree(self.elements, self.frequencies, self.pessimistic)
        else:
            log = "{0} deletion failed. N = {1}".format(key, len(self.elements))
        # with open("RandomOrderDynamic/log.txt", 'a') as writer:
        #     writer.write(log + "\n")


    """
    Recursively construct a height-balanced BST from the given elements.

    Parameters:
    - elements : list of orderable keys (duplicates not expected)

    Returns:
    - tuple (root, height) where
        - root   : BTNode or None
        - height : int (0 if no elements)

    Notes:
    - Sorts 'elements' and uses the median as root to balance subtrees.
    - Sets parent links for all children.
    """
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


    """
    Construct an Optimal BST (OBST) using classic O(n^3) DP.

    Parameters:
    - ordered_elements : list of keys in strictly sorted order
    - freq : list of access frequencies aligned to 'ordered_elements'

    Returns:
    - tuple (root, height) from get_optimal_tree_from_roots()

    Details:
    - Builds 'cost' and 'root' DP tables over all subarrays.
    - For each interval [i..j], tries each k in [i..j] as root and
        computes: cost[i][k-1] + cost[k+1][j] + sum(freq[i..j]).
    - Uses sum(...) directly (no prefix sums), hence O(n^3) total.
    """
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



    """
    Reconstruct a BST from a DP root table.

    Parameters:
    - ordered_elements : list of sorted keys
    - roots : 2D table where roots[i][j] stores the chosen root index for [i..j]
    - i, j : int bounds (inclusive) for the subarray to materialize

    Returns:
    - tuple (root, height) where
        - root   : BTNode or None
        - height : int (0 if empty interval)

    Notes:
    - Recursively builds left/right subtrees and sets parent pointers.
    """
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



    """
    Search for a key in the BST and track the number of steps.

    Parameters:
    - key_Value : the key to search for
    - __splay_cost__ : (unused here) compatibility flag for splay experiments

    Returns:
    - (node, cost) if found:
        - node : BTNode whose value == key_Value
        - cost : int number of node visits
    - None if not found (prints a diagnostic list of live keys)

    Notes:
    - Iterative BST search from self.root.
    """
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



    """
    Perform a breadth-first (level-order) traversal.

    Returns:
    - levels : list of lists of BTNode
        - levels[0] = [root]
        - levels[1] = nodes at depth 2, etc.

    Notes:
    - Nodes without children simply contribute no entries at the next level.
    """
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



    """
    Construct an Optimal BST using memoized DP with precomputed frequency sums.

    Parameters:
    - ordered_elements : list of keys in sorted order
    - freq : list of frequencies aligned with 'ordered_elements'

    Returns:
    - tuple (root, height) for the resulting OBST

    Details:
    - Precomputes fsum[i][j] = sum(freq[i..j]) to avoid O(n) segment sums.
    - Calls optCost_memoized(...) to fill DP tables.
    - Asymptotic time remains O(n^3) in the worst case but with lower constants.
    """
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



    """
    Convenience wrapper to build an OBST from unsorted elements + frequencies.

    Parameters:
    - elements : list of keys (not necessarily sorted)
    - frequencies : list of frequencies aligned with 'elements'

    Returns:
    - tuple (root, height) for the resulting OBST

    Behavior:
    - Zips elements with frequencies, sorts by key to get ordered lists.
    - Precomputes fsum and calls calc_optCost(...) which drives optCost_memoized.
    """
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



    """
    Bottom-up driver that fills the DP tables using the memoized routine.

    Parameters:
    - n : number of keys
    - freq : list of frequencies
    - cost : 2D DP table (mutated in-place)
    - roots : 2D DP table storing optimal root indices (mutated in-place)
    - freq_sum : 2D table where freq_sum[i][j] = sum(freq[i..j])

    Returns:
    - tuple (min_cost, cost, roots) where
        - min_cost : cost[0][n-1] (optimal expected search cost for all keys)
        - cost     : filled cost table
        - roots    : filled root table

    Notes:
    - Iterates i from n-1 down to 0 and j from i..n-1, calling optCost_memoized.
    """
    def calc_optCost(self, n, freq, cost, roots, freq_sum):
        i = n - 1
        while i >= 0:
            for j in range(i, n):
                c, cost, roots = self.optCost_memoized(freq, i, j, cost, roots, freq_sum)
            i -= 1
        return cost[0][n - 1], cost, roots


    """
    Memoized recursive routine to compute OBST cost for interval [i..j].

    Parameters:
    - freq : list of frequencies
    - i, j : inclusive bounds of the subproblem
    - cost : 2D DP table (mutated)
    - roots : 2D DP table of chosen roots (mutated)
    - freq_sum : 2D prefix-sum-like table, freq_sum[i][j] = sum(freq[i..j])

    Returns:
    - tuple (cost_ij, cost, roots) where
        - cost_ij : optimal cost for interval [i..j]

    Recurrence:
    - cost[i][j] = min over r in [i..j] of:
        cost[i][r-1] + cost[r+1][j] + freq_sum[i][j]

    Base Cases:
    - i > j  -> cost[i][j] = 0
    - cost[i][j] already computed -> return memoized value
    """
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
