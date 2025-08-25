from structures.BTNode import *
from structures.DS import *
import numpy as np

class Treap(DS):
    
    
    
    """
    Initialize a Treap data structure.

    Parameters:
    - elements : list of key values
    - frequencies : list of frequencies
    - opt : bool, whether to build optimal Treap (default True)
    - log_priority : bool, whether to use log-based randomized priority

    Notes: 
    - Initializes the Treap and generates the initial tree.
    """
    def __init__(self, elements, frequencies, opt=True, log_priority=True):
        DS.__init__(self)
        self.elements = elements
        self.frequencies = frequencies
        self.opt = opt                          # Whether to build optimal treap
        self.log_priority = log_priority
        self.root = None
        self.list_of_jobs = []                  # Used for incremental tree building
        self.node_count = len(elements) if elements else 0
        self.make_tree()                        # Build initial tree



    """
    Construct the Treap from current elements and frequencies.
    """
    def make_tree(self):
        element_freqs = []

        # Compute priorities for nodes
        for i in range(len(self.elements)):
            if self.log_priority:
                # Randomized priority using log2 of frequency
                priority = np.random.random() - np.log2(np.log2(1 / self.frequencies[i]))
            else:
                priority = self.frequencies[i]
            element_freqs.append([self.elements[i], priority])
        
        # Sort elements by priority (descending)
        sorted_elFreq = sorted(element_freqs, key=lambda t: t[1], reverse=True)  # sort by frequencies

        # Build tree based on 'opt' flag
        if not self.opt:
            self.root = self.__initial_tree(sorted_elFreq)
        else:
            # Make nodes, store jobs, and build tree incrementally
            sorted_nodes = self.make_sorted_nodes(sorted_elFreq)
            self.root = sorted_nodes[0] if len(sorted_nodes) > 0 else None
            self.list_of_jobs = [sorted_nodes]
            self.__initial_tree_opt()



    """
    Insert a new key into the Treap.

    Parameters:
    - key : key to insert
    - frequency : frequency value

    Rebuilds the Treap after insertion.
    """
    def insert(self, key, frequency):
        if key not in self.elements:
            self.elements.append(key)
            self.node_count += 1
            self.frequencies = list(self.frequencies)
            self.frequencies.append(frequency)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.make_tree()

    def get_node_count(self):
        return self.node_count



    """
    Delete a key from the Treap.

    Parameters:
    - key : key to remove

    Rebuilds the Treap after deletion.
    """
    def delete(self, key):
        if key in self.elements:
            idx = self.elements.index(key)
            self.elements.remove(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.pop(idx)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.node_count -= 1
            self.make_tree()



    """
    Recursively build a simple Treap from elements and frequencies.

    Parameters:
    - elements_frequencies : list of [value, priority] pairs

    Returns:
    - root node of subtree
    """
    def __initial_tree(self, elements_frequencies):
        if len(elements_frequencies) == 0:
            return None
        left_elFreq = []
        right_elFreq = []

        root = BTNode(elements_frequencies[0][0], freq=elements_frequencies[0][1])
        
        # Split remaining nodes into left and right based on root value
        for i in range(1, len(elements_frequencies)):
            if elements_frequencies[i][0] < root.value:
                left_elFreq.append(elements_frequencies[i])
            else:
                right_elFreq.append(elements_frequencies[i])
        
        left_root = self.__initial_tree(left_elFreq)
        right_root = self.__initial_tree(right_elFreq)

        root.left = left_root
        root.right = right_root

        # Set parent pointers
        if left_root is not None:
            left_root.parent = root
        if right_root is not None:
            right_root.parent = root

        return root



    """
    Convert list of [value, priority] to BTNode objects.

    Returns:
    - list of BTNode objects
    """
    def make_sorted_nodes(self, sorted_elFreqs):
        sorted_nodes = []
        for elfreq in sorted_elFreqs:
            node = BTNode(elfreq[0], freq=elfreq[1])
            sorted_nodes.append(node)
        return sorted_nodes



    """
    Build Treap incrementally using job list for optimal structure.
    
    Notes:
    - Calls __process_job until all nodes are attached.
    """
    def __initial_tree_opt(self):
        while(len(self.list_of_jobs)) > 0:
            self.__process_job()



    """
    Process a single list of sorted nodes to construct subtree.
    """
    def __process_job(self):

        if len(self.list_of_jobs) == 0:
            return
        left_nodes = []
        right_nodes = []

        # Pop the first job
        sorted_nodes = self.list_of_jobs[0]
        self.list_of_jobs = self.list_of_jobs[1:]

        root = sorted_nodes[0]
        
        for i in range(1, len(sorted_nodes)):
            if sorted_nodes[i].value < root.value:
                left_nodes.append(sorted_nodes[i])
            else:
                right_nodes.append(sorted_nodes[i])
        
        left_root = left_nodes[0] if len(left_nodes) > 0 else None
        right_root = right_nodes[0] if len(right_nodes) > 0 else None

        root.left = left_root
        root.right = right_root

        if left_root is not None:
            left_root.parent = root
        if right_root is not None:
            right_root.parent = root
        
        # Append remaining sublists as new jobs
        if len(left_nodes) > 0:
            self.list_of_jobs.append(left_nodes)
        if len(right_nodes) > 0 :
            self.list_of_jobs.append(right_nodes)

        return



    """
    Search for a key in the Treap.

    Parameters:
    - key_Value : key to search for

    Returns:
    - tuple (node, cost) if found, None if not found
    """
    def search(self, key_Value, __splay_cost__=False):
        node = self.root
        cost = 1
        while node.value != key_Value:
            if key_Value < node.value:
                node = node.left
                cost += 1
            else:
                node = node.right
                cost += 1

            if node is None:
                print("Couldn't find key {0}".format(key_Value))
                sorted_keys = sorted(self.elements.copy())
                print("List of live keys: ", sorted_keys)
                return None

        return node, cost



    """
    Breadth-first traversal of the Treap.

    Returns:
    - list of levels, each level is a list of nodes
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
