class BTNode():
    
    # Initialize a BTNode.
    def __init__(self, value, name="", freq=1):
        self.value = value      # Data
        self.name = name                    
        self.parent = None                  
        self.left = None                    
        self.right = None                   
        self.height = 1         # Node height (for balanced trees)
        self.depth = 1          # Node depth (distance from root)
        self.freq = freq        # Frequency counter (how many times this value appears)

    # Setters
    def set_parent(self, parent):
        self.parent = parent

    def set_left_child(self, child):
        self.left = child

    def set_right_child(self, child):
        self.right = child
