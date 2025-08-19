import random

class Node:
    def __init__(self, key, level):
        self.key = key
        # forward pointers (array of length = level + 1)
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, p=0.5):
        self.header = Node(float("-inf"), 0)  # start with level 0 header
        self.level = 0  # current max level of the skip list
        self.p = p      # probability for coin flips

    def random_level(self):
        lvl = 0
        while random.random() < self.p:
            lvl += 1
        return lvl

    def insert(self, key):
        update = [None] * (self.level + 1)
        node = self.header

        # find insertion position
        for i in reversed(range(self.level + 1)):
            while node.forward[i] and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node

        node = node.forward[0]

        if node is None or node.key != key:
            lvl = self.random_level()

            # if new node has higher level, grow header
            if lvl > self.level:
                for i in range(self.level + 1, lvl + 1):
                    self.header.forward.append(None)
                    update.append(self.header)
                self.level = lvl

            # create new node
            new_node = Node(key, lvl)

            # rewire pointers
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
                print(f"   → moved right to {node.key}")
            if node.forward[i] and node.forward[i].key == key:
                return node.forward[i], cost + 1
            # print(f"   ↓ drop down from level {i}")

        return None, cost

    def print_list(self):
        print("SkipList Levels:")
        for i in reversed(range(self.level + 1)):
            node = self.header.forward[i]
            level_nodes = []
            while node:
                level_nodes.append(str(node.key))
                node = node.forward[i]
            print(f"Level {i}: " + " -> ".join(level_nodes))



if __name__ == "__main__":
    sl = SkipList()
    nums = [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]
    for num in nums:
        sl.insert(num)
    
    sl.print_list()
    
    for key in [3, 9, 19, 100]:
        node, cost = sl.search(key)
        if node:
            print(f"Found {key} at cost ~{cost} steps")
        else:
            print(f"{key} not found (search cost ~{cost} steps)")
