

class SkipListNode:
    def __init__(self, value, name="", height=0):
        self.value = value
        self.name = name
        self.left = None
        self.right = None
        self.down = None
        self.top = None
        self.height = height

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

    def generate_down(self):
        down = SkipListNode(self.value, name=self.name)
        self.down = down

    def generate_top(self):
        top = SkipListNode(self.value, name=self.name, height=self.height)
        top.down = self
        self.top = top
        return top