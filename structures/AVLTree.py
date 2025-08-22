import sys

from BTNode import *
from DS import *


class AVLTree(DS):

    def __init__(self, elements=[]):
        DS.__init__(self)
        self.root = None
        self.elements = []
        if len(elements) > 0:
            for v in elements:
                self.insert(v, None)

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

    def insert(self, key, freq):
        if key not in self.elements:
            self.elements.append(key)
            log = "{0} inserted. N = {1}".format(key, len(self.elements))
        else:
            log = "{0} insertion failed. N = {1}".format(key, len(self.elements))
        # with open("RandomOrderDynamic/log.txt", 'a') as writer:
        #     writer.write(log + "\n")
        node = self.__insert_node(self.root, key)

        self.root = node
        return node

    # Function to insert a node
    def __insert_node(self, root, key):

        # Find the correct location and insert the node
        if not root:
            return BTNode(key)
        elif key < root.value:
            root.left = self.__insert_node(root.left, key)
        else:
            root.right = self.__insert_node(root.right, key)

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        # Update the balance factor and balance the tree
        balanceFactor = self.getBalance(root)
        if balanceFactor > 1:
            if key < root.left.value:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

        if balanceFactor < -1:
            if key > root.right.value:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        return root

    def delete(self, key):
        if key in self.elements:
            self.elements.remove(key)
            log = "{0} deleted. N = {1}".format(key, len(self.elements))
        else:
            log = "{0} deletion failed. N = {1}".format(key, len(self.elements))
        # with open("RandomOrderDynamic/log.txt", 'a') as writer:
        #     writer.write(log + "\n")
        self.root = self.__delete_node(self.root, key)

    # Function to delete a node
    def __delete_node(self, root, key):

        # Find the node to be deleted and remove it
        if not root:
            return root
        elif key < root.value:
            root.left = self.__delete_node(root.left, key)
        elif key > root.value:
            root.right = self.__delete_node(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                self.root = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                self.root = root.left
                root = None
                return temp
            temp = self.getMinValueNode(root.right)
            root.value = temp.value
            root.name = temp.name
            root.right = self.__delete_node(root.right,
                                            temp.value)
        if root is None:
            return root

        # Update the balance factor of nodes
        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)

        # Balance the tree
        if balanceFactor > 1:
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    # Function to perform left rotation
    def leftRotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    # Function to perform right rotation
    def rightRotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    # Get the height of the node
    def getHeight(self, root):
        if not root:
            return 0
        return root.height

    # Get balance factore of the node
    def getBalance(self, root):
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMinValueNode(self, root):
        if root is None or root.left is None:
            return root
        return self.getMinValueNode(root.left)

    def preOrder(self, root):
        if not root:
            return
        print("{0} ".format(root.value), end="")
        self.preOrder(root.left)
        self.preOrder(root.right)

    # Print the tree
    def printHelper(self, currPtr, indent, last):
        if currPtr != None:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "
            print(currPtr.value)
            self.printHelper(currPtr.left, indent, False)
            self.printHelper(currPtr.right, indent, True)
