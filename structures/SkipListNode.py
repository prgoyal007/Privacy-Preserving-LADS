class SkipListNode:
    
    # Initialize a SkipList node.
    def __init__(self, value, name="", height=0):
        self.value = value          # data
        self.name = name
        self.left = None
        self.right = None
        self.down = None
        self.top = None
        self.height = height        # layer/height

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

    """
    Generate a downward node (new node at the level below).
    Useful when building links between layers.
    """
    def generate_down(self):
        down = SkipListNode(self.value, name=self.name)
        self.down = down

    """
    Generate an upward node (new node at the level above).
    This links the current node to a new node one level higher.
        
    Returns:
    - SkipListNode: the newly created top node
    """
    def generate_top(self):
        top = SkipListNode(self.value, name=self.name, height=self.height)
        top.down = self
        self.top = top
        return top