from BinaryTree import *
from SkipListNode import *
from numpy import *

class DynamicRSL(DS):
    def __init__(self, ordered_elements, frequencies, p0=0.9, pg=0.368, alpha=2, right_comparison=False):
        DS.__init__(self)
        self.pg = pg
        self.c = 1  # TODO
        self.beta = 4 / np.log2(1 / self.pg)  # TODO
        self.alpha = alpha  # TODO
        self.HFactor = 50  # TODO
        self.p0 = p0  # TODO
        self.memory = 0
        self.right_comparison = right_comparison
        self.setup(ordered_elements, frequencies)

        # self.nStar = len(ordered_elements)
        # self.n = 4
        # while self.n < self.nStar:
        #     self.n *= self.n
        # self.elements = ordered_elements
        # self.frequencies = frequencies
        # self.first_node, self.last_node, self.height = self.make_skip_list(ordered_elements, frequencies)





    def setup(self, ordered_elements, frequencies):
        self.nStar = len(ordered_elements)
        self.n = 4
        while self.n < self.nStar:
            self.n *= self.n
        self.elements = ordered_elements
        self.frequencies = frequencies
        # self.pg = pg
        # self.c = 1  # TODO
        # self.beta = 4 / np.log2(1 / self.pg)  # TODO
        # self.alpha = alpha  # TODO
        # self.HFactor = 50  # TODO
        # self.p0 = p0  # TODO
        self.first_node, self.last_node, self.height = self.make_skip_list(ordered_elements, frequencies)
        # self.right_comparison = right_comparison

    def pMin(self, i):
        return self.p0 ** (2**i)

    def delta_i(self, i):
        return np.log2(self.alpha * self.pMin(i)) / np.log2(self.pg)

    def get_height(self, fi, H):
        X = 0
        while True:
            if np.random.random() < self.pg:
                X += 1
            else:
                break
        return min(H, np.ceil(H - fi + X))

    def get_fValues(self, k):
        fVals = []
        for i in range(k):
            f = self.delta_i(k+1)
            if fVals:
                f += fVals[-1]
            fVals.append(f)
        return fVals


    def get_i(self, frequency):
        p = self.p0
        i = 0
        while frequency < p:
            p *= p
            i += 1
        return i







    def make_skip_list(self, ordered_elements, frequencies):
        k = max(2, self.get_i(self.n**(-self.c/self.beta)) + 1)
        H = int(self.HFactor * np.log2(self.n))

        hValues = [float('inf')]
        fValues = self.get_fValues(k)
        for freq in frequencies:
            i = self.get_i(max(freq, self.n**(-self.c/self.beta)))
            fVal = fValues[i]
            hValues.append(self.get_height(fVal, H))
        hValues.append(float('inf'))
        # updating H to have the lowest hieght equal to zero
        hMin = min(hValues)
        for i in range(len(hValues)):
            hValues[i] -= hMin

        #generating the bottom layer
        nodes = [SkipListNode(float('-inf'), height=float('inf'))]
        last_layer_node = nodes[0]
        for i, e in enumerate(ordered_elements):
            n = SkipListNode(e, height=hValues[i+1])
            self.memory += hValues[i+1] + 1
            nodes[-1].right = n
            n.left = nodes[-1]
            nodes.append(n)
        n = SkipListNode(float('inf'), height=float('inf'))
        nodes[-1].right = n
        n.left = nodes[-1]
        nodes.append(n)
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


    def insert(self, key, freq):
        # self.actual_insert(key, freq)
        self.fake_insert(key, freq)

    def fake_insert(self, key, freq):
        if key not in self.elements:
            self.elements.append(key)
            self.elements = sorted(self.elements)
            idx = self.elements.index(key)
            self.frequencies.insert(idx, freq)
            self.setup(self.elements, self.frequencies)

    def actual_insert(self, key, freq):
        if key not in self.elements:
            self.elements.append(key)
            self.elements = sorted(self.elements)
            idx = self.elements.index(key)
            self.frequencies.insert(idx, freq)
            self.nStar += 1
            if self.nStar >= self.n:
                self.n *= self.n
                # print("<< N got doubled >>")
                self.first_node, self.last_node, self.height = self.make_skip_list(self.elements, self.frequencies)
                return
            else:
                node = self.last_node
                while node.right.value <= key:
                    node = node.right
                i = self.get_i(max(freq, self.n ** (-self.c / self.beta)))
                k = max(2, self.get_i(self.n**(-self.c/self.beta)) + 1)
                fValues = self.get_fValues(k)
                fVal = fValues[i]
                hNode = max(self.get_height(fVal, int(self.HFactor * np.log2(self.n))), 0)
                hNode = min(hNode, self.height + 1)
                if hNode > self.height:
                    self.height += 1
                    first_top = self.first_node.generate_top()
                    tmpNode = self.first_node
                    while tmpNode.right is not None:
                        tmpNode = tmpNode.right
                    inf_top = tmpNode.generate_top()
                    first_top.right = inf_top
                    inf_top.left = first_top
                    self.first_node=  first_top


                h = 0
                newNode = None
                while True:
                    if newNode is None:
                        newNode = SkipListNode(key, height=hNode)
                    newNode.right = node.right
                    newNode.right.left = newNode
                    node.right = newNode
                    newNode.left = node
                    h += 1
                    if h >= hNode:
                        break
                    while node.top is None:
                        node = node.left
                    node = node.top
                    newNode = newNode.generate_top()





    def delete(self, key):
        if key in self.elements:
            node, c = self.search(key)
            idx = self.elements.index(key)
            self.elements.remove(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.pop(idx)
            self.nStar -= 1
            if self.nStar <= np.sqrt(np.sqrt(self.n)):
                self.n = np.ceil(np.sqrt(self.n))
                print("<< N got squared >>")
                self.n = max(self.n, 2)
                self.first_node, self.last_node, self.height = self.make_skip_list(self.elements, self.frequencies)
                return
            while node is not None:
                left = node.left
                right = node.right
                left.right = right
                right.left = left
                node = node.down
            while self.first_node.right.value == float('inf') and self.first_node.down is not None:
                self.first_node = self.first_node.down
                self.height -= 1


    def search(self, key_Value, __splay_cost__=False):
        cost = 0
        node = self.first_node
        upper_bound = float('inf')
        while True:
            if node is None:
                return None, cost
            if node.value == key_Value:
                return node, cost
            if node.right.value <= key_Value:
                node = node.right
                cost += 1
                continue
            else:
                if node.right.value < upper_bound:
                    if self.right_comparison:
                        cost += 1
                    upper_bound = min(node.right.value, upper_bound)
                node = node.down

    def BFS(self, __print__=False):
        first_node = self.first_node
        levels = []
        for i in range(self.height):
            node = first_node
            level = []
            while node is not None:
                if __print__:
                    print("\t[ {0} ]".format(node.value), end="")
                level.append(node)
                node = node.right
            levels.append(level)
            first_node = first_node.down
            if __print__:
                print("")
        return levels
