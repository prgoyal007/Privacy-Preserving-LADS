import numpy as np
import builtins
from structures.BinaryTree import *
from structures.SkipListNode import *

class StaticRSL(DS):
    
    
    
    """
    Initialize the StaticRSL skip list with a predetermined static structure based on
    element ordering and frequencies.

    Parameters:
    - ordered_elements : list
        Elements in sorted order to be inserted into the skip list.
    - frequencies : list
        Access frequencies corresponding to each element.
    - p0 : float (default=0.9)
        Base probability used for determining minimum probability for levels.
    - pg : float (default=0.368)
        Geometric probability for delta calculations.
    - alpha : float (default=2)
        Scaling factor for delta calculations.
    - right_comparison : bool (default=False)
        Whether to count right comparisons during search.

    Side effects:
    - Sets self.first_node, self.last_node, and self.height via make_skip_list().
    """
    def __init__(self, ordered_elements, frequencies, p0=0.9, pg=0.368, alpha=2, right_comparison=False):
        DS.__init__(self)
        self.pg = pg
        self.alpha = alpha 
        self.HFactor = 50  
        self.p0 = p0                                           # Set to high frequency = 0.9
        self.right_comparison = right_comparison

        # Store elements and frequencies
        self.elements = ordered_elements
        self.frequencies = frequencies

        # Build the skip list
        self.first_node, self.last_node, self.height = self.make_skip_list(ordered_elements, frequencies)



    """
    Compute minimum probability at level i.

    Parameters:
    - i : int
        Level index

    Returns:
    - float : p0 raised to the 2^i power
    """
    def pMin(self, i):
        return self.p0 ** (2**i)



    """
    Compute delta_i used in static skip list height calculations.

    Parameters:
    - i : int
        Index used to compute delta_i

    Returns:
    - float : delta value for level i
    """
    def delta_i(self, i):
        return np.log2(self.alpha * self.pMin(i)) / np.log2(self.pg)


    """
    Determine static level index for a given frequency.

    Parameters:
    - frequency : float
        Frequency value used to calculate level

    Returns:
    - int : computed level index i
    """
    def get_i(self, frequency):
        p = self.p0
        i = 0
        while frequency < p:
            p *= p
            i += 1
        return i



    """
    Compute the cumulative f-values for k levels.

    Parameters:
    - k : int
        Number of levels

    Returns:
    - list of floats : cumulative f-values for each level
    """
    def get_fValues(self, k):
        fVals = []
        for i in range(k):
            f = self.delta_i(k+1)
            if fVals:
                f += fVals[-1]
            fVals.append(f)
        return fVals



    """
    Compute the fixed height for a node based on f-value.

    Parameters:
    - fi : float
        Delta/f-value for the node
    - H : int
        Maximum allowed height

    Returns:
    - int : final height of the node
    """
    def get_height(self, fi, H):
        # Heights are fixed; no dynamic adjustment
        return min(H, np.ceil(H - fi))                                                      



    """
    Construct the static skip list with fixed node heights.

    Parameters:
    - ordered_elements : list
        Elements in sorted order
    - frequencies : list
        Corresponding frequencies

    Returns:
    - tuple :
        first_node : bottom-left node of the skip list
        last_layer_node : last node in the bottom layer
        h : int, total number of levels in the skip list
    """
    def make_skip_list(self, ordered_elements, frequencies):
        n = len(ordered_elements)
        k = builtins.max(2, self.get_i(n**(-1/4)) + 1)                   # simplified static calculation
        H = int(self.HFactor * np.log2(n))

        hValues = [float('inf')]
        fValues = self.get_fValues(k)
        for freq in frequencies:
            i = self.get_i(freq)
            i = min(self.get_i(freq), len(fValues)-1)
            fVal = fValues[i]
            hValues.append(self.get_height(fVal, H))
        hValues.append(float('inf'))

        # Normalize so minimum height = 0
        hMin = min(hValues)
        hValues = [h - hMin for h in hValues]

        # Build bottom layer
        nodes = [SkipListNode(float('-inf'), height=float('inf'))]
        last_layer_node = nodes[0]
        for i, e in enumerate(ordered_elements):
            n = SkipListNode(e, height=hValues[i+1])
            nodes[-1].right = n
            n.left = nodes[-1]
            nodes.append(n)
        n = SkipListNode(float('inf'), height=float('inf'))
        nodes[-1].right = n
        n.left = nodes[-1]
        nodes.append(n)

        # Build upper layers (fixed, once)
        h = 0
        while len(nodes) > 2:
            next_layer = []
            for node in nodes:
                if node.height > h:
                    top = node.generate_top()
                    next_layer.append(top)
            for i in range(len(next_layer) - 1):
                node = next_layer[i]
                next_node = next_layer[i+1]                    
                node.right = next_node
                next_node.left = node
            nodes = next_layer
            h += 1
        
        first = nodes[0].down
        return first,last_layer_node, h



    """
    Static insertion: update elements/frequencies list without changing structure.

    Parameters:
    - key : element to insert
    - freq : float (optional)
        Frequency of the element

    Side effects:
    - Updates self.elements (sorted) and self.frequencies
    """
    def insert(self, key, freq=None):
        # Static insert: only update element/frequency lists
        if key not in self.elements:
            self.elements.append(key)
            self.elements = sorted(self.elements)
            idx = self.elements.index(key)
            if freq is not None:
                self.frequencies.insert(idx, freq)
        


    """
    Static deletion: remove element and frequency without changing structure.

    Parameters:
    - key : element to remove

    Side effects:
    - Updates self.elements and self.frequencies
    """
    def delete(self, key):
        # Static delete: only update element/frequency lists
        if key in self.elements:
            idx = self.elements.index(key)
            self.elements.pop(idx)
            self.frequencies.pop(idx)

    """
    Standard skip list search for a given key.

    Parameters:
    - key_Value : element to search
    - __splay_cost__ : bool (ignored, for API compatibility)

    Returns:
    - tuple :
        Node if found, else None
        int : cost (number of traversed nodes)
    """
    def search(self, key_Value, __splay_cost__=False):
        cost = 0
        node = self.first_node
        upper_bound = float('inf')
        while node:
            if node.value == key_Value:
                return node, cost
            if node.right.value <= key_Value:
                node = node.right
                cost += 1
            else:
                if node.right.value < upper_bound:
                    if self.right_comparison:
                        cost += 1
                    upper_bound = min(node.right.value, upper_bound)
                node = node.down
        return None, cost



    """
    Breadth-first traversal of the skip list.

    Parameters:
    - __print__ : bool
        If True, print the nodes at each level

    Returns:
    - list of lists of nodes : each sublist corresponds to a level
    """
    def BFS(self, __print__=False):
        first_node = self.first_node
        levels = []
        for _ in range(self.height):
            node = first_node
            level = []
            while node:
                if __print__:
                    print(f"\t[ {node.value} ]", end="")
                level.append(node)
                node = node.right
            levels.append(level)
            first_node = first_node.down
            if __print__:
                print("")
        return levels