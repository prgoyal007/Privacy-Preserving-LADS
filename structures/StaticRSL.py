from .BinaryTree import *
from .SkipListNode import *
from numpy import *

class StaticRSL(DS):
    def __init__(self, ordered_elements, frequencies, p0=0.9, pg=0.368, alpha=2, right_comparison=False):
        DS.__init__(self)
        self.pg = pg
        self.alpha = alpha 
        self.HFactor = 50  
        self.p0 = p0  
        self.right_comparison = right_comparison

        # Store elements and frequencies
        self.elements = ordered_elements
        self.frequencies = frequencies

        # Build the skip list
        self.first_node, self.last_node, self.height = self.make_skip_list(ordered_elements, frequencies)

    def pMin(self, i):
        return self.p0 ** (2**i)

    def delta_i(self, i):
        return np.log2(self.alpha * self.pMin(i)) / np.log2(self.pg)

    def get_i(self, frequency):
        p = self.p0
        i = 0
        while frequency < p:
            p *= p
            i += 1
        return i

    def get_fValues(self, k):
        fVals = []
        for i in range(k):
            f = self.delta_i(k+1)
            if fVals:
                f += fVals[-1]
            fVals.append(f)
        return fVals

    def get_height(self, fi, H):
        # Heights are fixed; no dynamic adjustment
        return min(H, np.ceil(H - fi))                                                      





    def make_skip_list(self, ordered_elements, frequencies):
        n = len(ordered_elements)
        k = max(2, self.get_i(n**(-1/4)) + 1)  # simplified static calculation
        H = int(self.HFactor * np.log2(n))

        hValues = [float('inf')]
        fValues = self.get_fValues(k)
        for freq in frequencies:
            i = self.get_i(freq)
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



    def insert(self, key, freq=None):
        # Static insert: only update element/frequency lists
        if key not in self.elements:
            self.elements.append(key)
            self.elements = sorted(self.elements)
            idx = self.elements.index(key)
            if freq is not None:
                self.frequencies.insert(idx, freq)
        
    def delete(self, key):
        # Static delete: only update element/frequency lists
        if key in self.elements:
            idx = self.elements.index(key)
            self.elements.pop(idx)
            self.frequencies.pop(idx)

    # Standard skip list search
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