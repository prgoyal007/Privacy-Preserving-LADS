import math
import random

class Node:
    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

class BiasedSkipList:
    def __init__(self, frequencies, capacity):
        self.capacity = capacity          # total number of elements
        self.max_level = int(math.log2(capacity)) if capacity > 1 else 1
        self.header = Node(float("-inf"), self.max_level)
        self.level = 0
        self.frequencies = frequencies    # dict: {key: frequency}

    def random_level(self, freq=None):
        # geometric part
        lvl = 0
        while random.random() < 0.5 and lvl < self.max_level:
            lvl += 1

        if freq is not None:
            # add bias: floor(log2(freq * n)) 
            # (ensure argument ≥ 1 before taking log2)
            biased = math.floor(math.log2(max(1, freq * self.capacity)))
            lvl += biased

        return min(lvl, self.max_level)

    def insert(self, key):
        freq = self.frequencies.get(key, 1 / self.capacity)
        update = [None] * (self.max_level + 1)
        node = self.header

        # find insert position
        for i in reversed(range(self.level + 1)):
            while node.forward[i] and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node

        node = node.forward[0]

        if not node or node.key != key:
            lvl = self.random_level(freq)
            if lvl > self.level:
                for i in range(self.level + 1, lvl + 1):
                    update[i] = self.header
                self.level = lvl

            new_node = Node(key, lvl)
            for i in range(lvl + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

    def search(self, key):
        node = self.header
        cost = 0
        # print(f"Searching for {key}...")

        for i in reversed(range(self.level + 1)):  # top to bottom
            # print(f" Level {i}: starting at {node.key if node else 'None'}")
            while node.forward[i] and node.forward[i].key < key:
                node = node.forward[i]
                cost += 1
                # print(f"   → moved right to {node.key}")
            if node.forward[i] and node.forward[i].key == key:
                return node.forward[i], cost + 1
            # print(f"   ↓ drop down from level {i}")

        return None, cost

    def display(self):
        print("\nSkipList Levels:")
        for i in reversed(range(self.level + 1)):
            node = self.header.forward[i]
            line = f"Level {i}: "
            while node:
                line += str(node.key) + " -> "
                node = node.forward[i]
            print(line[:-4])


if __name__ == "__main__":
    # Hardcoded test frequencies that sum to 1
    freqs = {
        "a": 0.25,
        "b": 0.20,
        "c": 0.15,
        "d": 0.10,
        "e": 0.10,
        "f": 0.08,
        "g": 0.07,
        "h": 0.05
    }

    n = len(freqs)
    bsl = BiasedSkipList(frequencies=freqs, capacity=n)

    for k in freqs.keys():
        bsl.insert(k)

    bsl.display()

    # Try searching for some keys
    for key in ["a", "d", "h", "z"]:
        node, cost = bsl.search(key)
        if node:
            print(f"Found {key} at cost {cost}")
        else:
            print(f"{key} not found, cost = {cost}")
