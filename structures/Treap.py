from BTNode import *
from DS import *
import numpy as np

class Treap(DS):
    def __init__(self, elements, frequencies, opt=True, log_priority=False):
        DS.__init__(self)
        self.elements = elements
        self.frequencies = frequencies
        self.opt = opt
        self.log_priority = log_priority
        self.root = None
        self.list_of_jobs = []
        self.make_tree()

    def make_tree(self):
        element_freqs = []
        for i in range(len(self.elements)):
            if self.log_priority:
                priority = np.random.random() - np.log2(np.log2(1 / self.frequencies[i]))
            else:
                priority = self.frequencies[i]
            element_freqs.append([self.elements[i], priority])
        sorted_elFreq = sorted(element_freqs, key=lambda t: t[1], reverse=True)  # sort by frequencies

        if not self.opt:
            self.root = self.__initial_tree(sorted_elFreq)
        else:
            sorted_nodes = self.make_sorted_nodes(sorted_elFreq)
            self.root = sorted_nodes[0] if len(sorted_nodes) > 0 else None
            self.list_of_jobs = [sorted_nodes]
            self.__initial_tree_opt()

    def insert(self, key, frequency):
        if key not in self.elements:
            self.elements.append(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.append(frequency)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.make_tree()


    def delete(self, key):
        if key in self.elements:
            idx = self.elements.index(key)
            self.elements.remove(key)
            self.frequencies = list(self.frequencies)
            self.frequencies.pop(idx)
            self.frequencies = np.array(self.frequencies) / np.sum(self.frequencies)
            self.make_tree()

    def __initial_tree(self, elements_frequencies):
        """

        :param elements_frequencies: array of tuples: (element value, element frequency)
        :return:
        """
        if len(elements_frequencies) == 0:
            return None
        left_elFreq = []
        right_elFreq = []

        root = BTNode(elements_frequencies[0][0], freq=elements_frequencies[0][1])
        for i in range(1, len(elements_frequencies)):
            if elements_frequencies[i][0] < root.value:
                left_elFreq.append(elements_frequencies[i])
            else:
                right_elFreq.append(elements_frequencies[i])
        left_root = self.__initial_tree(left_elFreq)
        right_root = self.__initial_tree(right_elFreq)

        root.left = left_root
        root.right = right_root

        if left_root is not None:
            left_root.parent = root
        if right_root is not None:
            right_root.parent = root

        return root

    def make_sorted_nodes(self, sorted_elFreqs):
        sorted_nodes = []
        for elfreq in sorted_elFreqs:
            node = BTNode(elfreq[0], freq=elfreq[1])
            sorted_nodes.append(node)
        return sorted_nodes

    def __initial_tree_opt(self):
        while(len(self.list_of_jobs)) > 0:
            self.__process_job()

    def __process_job(self):

        if len(self.list_of_jobs) == 0:
            return
        left_nodes = []
        right_nodes = []
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
        if len(left_nodes) > 0:
            self.list_of_jobs.append(left_nodes)
        if len(right_nodes) > 0 :
            self.list_of_jobs.append(right_nodes)


        return

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
