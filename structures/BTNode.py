class BTNode():
    def __init__(self, value, name="", freq=1):
        self.value = value
        self.name = name
        self.parent = None
        self.left = None
        self.right = None
        self.height = 1
        self.depth = 1
        self.color = 1
        self.freq = freq

    def set_parent(self, parent):
        self.parent = parent

    def set_left_child(self, child):
        self.left = child

    def set_right_child(self, child):
        self.right = child
