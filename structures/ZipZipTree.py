# explanations for member functions are provided in requirements.py
# each file that uses a Zip Tree should import it from this file

from __future__ import annotations

from typing import TypeVar
from dataclasses import dataclass
import random
import math

KeyType = TypeVar('KeyType')
ValType = TypeVar('ValType')

@dataclass
class Rank:
	geometric_rank: int
	uniform_rank: int

class ZipZipTree:
	
	
	
	"""
	Initialize a ZipZipTree.

	Parameters:
	- capacity : Maximum number of nodes expected (affects rank scaling)
	"""
	def __init__(self, capacity: int):
		self.capacity = capacity
		self.root = None
		self.size = 0	# number of nodes in the tree



	"""
    Generate a random rank for a node based on geometric and uniform distributions.    

    Returns:
    - Rank object (geometric_rank, uniform_rank)
	"""
	def get_random_rank(self) -> Rank:
		# Geometric Distribution: # of failures before the first success
		geo_rank = 0
		while random.random() < 0.5:
			geo_rank += 1
		
		# Uniform Distribution: random int from 0 to log(capacity)^3 - 1
		uniform_max = int(math.log(self.capacity, 2) ** 3) - 1
		uniform_rank = random.randint(0, max(uniform_max, 0))

		return Rank(geo_rank, uniform_rank)



	"""
	Insert a new key-value pair into the ZipZipTree, optionally using a given rank.

	Parameters:
	- key : KeyType
		The key of the new node to insert.
	- val : ValType
		The value associated with the key.
	- rank : Rank (optional)
		Precomputed rank to use instead of generating a random one.

	Side effects:
	- Updates self.root to reflect the inserted node.
	- Increments self.size by 1.
	"""
	def insert(self, key: KeyType, val: ValType, rank: Rank = None):
		# Start by generating a random rank if not provided
		if rank is None:
			rank = self.get_random_rank()	# generate a random rank in form of a tuple (geometric_rank, uniform_rank)
		
		new_node = Node(key, val, rank)	# create a new node with the given key, value, and rank

		self.root = self.insert_node(self.root, new_node)	# insert the new node into the tree
		self.size += 1	# increment the size of the tree



	"""
	Recursively insert a node into the subtree, maintaining Zip Tree properties.

	Parameters:
	- node : Node
		Root of the current subtree.
	- new_node : Node
		Node to insert.

	Returns:
	- Node : New root of the subtree after insertion.

	Side effects:
	- May modify left/right child pointers of nodes along the path.
	"""
	def insert_node(self, node: Node, new_node: Node) -> Node:
		if not node:
			return new_node
		
		if self.is_higher_rank(new_node.rank, node.rank, new_node.key, node.key):
			# Unzip the tree into two trees based on the new node's key and rank
			left, right = self.unzip(node, new_node.key, new_node.rank)
			new_node.left = left
			new_node.right = right
			return new_node
		
		# Recurse
		if new_node.key < node.key:
			node.left = self.insert_node(node.left, new_node)
		else:
			node.right = self.insert_node(node.right, new_node)

		return node	# unchanged root



	"""
	Remove a node with a given key from the ZipZipTree.

	Parameters:
	- key : KeyType
		The key of the node to remove.

	Side effects:
	- Updates self.root to reflect the removal.
	- Decrements self.size by 1.
	"""
	def remove(self, key: KeyType):
		self.root = self.remove_node(self.root, key)	# remove the node with the given key from the tree
		self.size -= 1	# decrement the size of the tree
	

	
	"""
	Recursively remove a node from a subtree, maintaining Zip Tree properties.

	Parameters:
	- node : Node
		Root of the current subtree.
	- key : KeyType
		Key of the node to remove.

	Returns:
	- Node : New root of the subtree after removal.

	Raises:
	- KeyError if the key is not found.

	Side effects:
	- May modify left/right child pointers of nodes along the path.
	"""
	def remove_node(self, node: Node, key: KeyType) -> Node:
		if not node:
			raise KeyError(f"Key {key} not found in the tree.")

		if key < node.key:
			node.left = self.remove_node(node.left, key)
		elif key > node.key:
			node.right = self.remove_node(node.right, key)
		else:
			# key == node.key, found the node to remove
			return self.zip(node.left, node.right)	# zip the left and right subtrees; returns the new root of the subtree
		
		return node	# unchanged root



	"""
	Find the value associated with a key.

	Parameters:
	- key : KeyType
		The key to search for.

	Returns:
	- Value associated with the key if found, else None.
	"""
	def find(self, key: KeyType):
		node = self.root
		while node:
			if key == node.key:
				# found the key, return the value
				return node.val
			elif key < node.key:
				node = node.left
			else:
				node = node.right
		# key not found in the tree
		return None
	


	"""
	Find a value by key and track the number of comparisons performed.

	Parameters:
	- key : KeyType
		The key to search for.

	Returns:
	- Tuple (value, cost) where:
		- value : value associated with key (None if not found)
		- cost : number of comparisons performed
	"""
	def find_with_cost(self, key: KeyType):
		node = self.root
		cost = 0
		while node:
			cost += 1
			if key == node.key:
				# found the key, return the value
				return node.val, cost
			elif key < node.key:
				node = node.left
			else:
				node = node.right
			# key not found in the tree
		return None, cost



	"""
	Return the current number of nodes in the ZipZipTree.

	Returns:
	- int : number of nodes in the tree (self.size)
	"""
	def get_size(self) -> int:
		return self.size



	"""
	Compute the height of the ZipZipTree.

	Returns:
	- int : height of the tree (max depth from root to any leaf)	
	"""
	def get_height(self) -> int:
		return self.recurse_height(self.root)



	"""
	Compute the depth of a node with the given key.

	Parameters:
	- key : KeyType
		The key of the node to measure depth.

	Returns:
	- int : depth of the node (distance from root)

	Raises:
	- KeyError if the key is not found.
	"""
	def get_depth(self, key: KeyType):
		# similar to get_height, but only for the path from the root to the node with the given key
		return self.recurse_depth(self.root, key, 0)



	"""
	Recursively compute the height of a subtree.

	Parameters:
	- node : Node
		Root of the current subtree.

	Returns:
	- int : height of the subtree
	"""
	def recurse_height(self, node: Node) -> int:
		if not node:
			return -1
		else:
			return 1 + max(self.recurse_height(node.left), self.recurse_height(node.right))
	


	"""
	Recursively compute the depth of a node in a subtree.

	Parameters:
	- node : Node
		Root of the current subtree.
	- key : KeyType
		The key of the node whose depth is being computed.
	- current_depth : int
		Depth accumulated along the recursion.

	Returns:
	- int : depth of the node with the given key

	Raises:
	- KeyError if the key is not found.
	"""
	def recurse_depth(self, node: Node, key: KeyType, current_depth: int) -> int:
		if not node:
			raise KeyError(f"Key {key} not found in the tree.")
		
		if key == node.key:
			return current_depth
		elif key < node.key:
			return self.recurse_depth(node.left, key, current_depth + 1)
		else:
			return self.recurse_depth(node.right, key, current_depth + 1)
	


	"""
	Compare two ranks to determine which node has higher priority.

	Parameters:
	- rank1 : Rank
		Rank of the first node.
	- rank2 : Rank
		Rank of the second node.
	- k1 : KeyType (optional)
		Key of the first node (used for tie-breaking).
	- k2 : KeyType (optional)
		Key of the second node (used for tie-breaking).

	Returns:
	- bool : True if rank1 is higher than rank2, else False
	"""
	def is_higher_rank(self, rank1: Rank, rank2: Rank, k1: KeyType = None, k2: KeyType = None) -> bool:
		# Compare the ranks based on the geometric rank first, then the uniform rank
		if rank1.geometric_rank != rank2.geometric_rank:	# compare geometric ranks first
			return rank1.geometric_rank > rank2.geometric_rank
		if rank1.uniform_rank != rank2.uniform_rank:	# compare uniform ranks next
			return rank1.uniform_rank > rank2.uniform_rank
		return k1 < k2 	# tie-breaker based on the keys if ranks are equal
		
	

	"""
	Split (unzip) a subtree into two trees based on a given key and rank.

	Parameters:
	- node : Node
		Root of the current subtree to unzip.
	- key : KeyType
		Key of the node being inserted that may become the new root.
	- rank : Rank
		Rank of the node being inserted.

	Returns:
	- tuple (Node, Node) : left and right subtrees after unzipping
	"""
	def unzip(self, node: Node, key: KeyType, rank: Rank) -> tuple[Node, Node]:
		# Unzip the tree into two trees based on the given key and rank
		if not node:
			return None, None
		
		if self.is_higher_rank(rank, node.rank, key, node.key):
			# If the rank is higher, unzip the left and right subtrees
			if key < node.key:
				left, right = self.unzip(node.left, key, rank)
				node.left = right
				return left, node
			else:
				left, right = self.unzip(node.right, key, rank)
				node.right = left
				return node, right
		else:
			# Stop unzipping if the rank is not higher
			if key < node.key:
				# If key is less than node's key, unzip the left subtree
				return None, node
			else:
				# If key is greater than node's key, unzip the right subtree
				return node, None
	


	"""
	Merge (zip) two subtrees into one, maintaining Zip Tree properties.

	Parameters:
	- left : Node
		Root of the left subtree.
	- right : Node
		Root of the right subtree.

	Returns:
	- Node : new root of the merged subtree
	"""
	def zip(self, left: Node, right: Node) -> Node:
		# Zip two trees together based on their ranks

		# Base case: if either left or right is None, return the other
		if not left:
			return right
		if not right:
			return left
		
		# Zip the two trees together, choose the root based on the ranks
		if self.is_higher_rank(left.rank, right.rank, left.key, right.key):
			left.right = self.zip(left.right, right)
			return left
		else:
			right.left = self.zip(left, right.left)
			return right



	"""
	Pretty-print the tree structure with keys and ranks.

	Parameters:
	- node : Node (optional)
		Root of the subtree to print; defaults to tree root.
	- indent : str
		Indentation string for formatting.
	- position : str
		Position label for current node ('root', 'L', 'R').
	- visited : set
		Tracks visited nodes to avoid cycles.

	Side effects:
	- Prints the tree to stdout.
	"""
	def print_tree(self, node=None, indent="", position="root", visited=None):
		if visited is None:
			visited = set()
		if node is None:
			node = self.root
		if not node or id(node) in visited:
			return

		visited.add(id(node))
		print(f"{indent}[{position}] key={node.key}, rank=({node.rank.geometric_rank}, {node.rank.uniform_rank})")
		self.print_tree(node.left, indent + "  ", "L", visited)
		self.print_tree(node.right, indent + "  ", "R", visited)



class Node:
		def __init__(self, key, val, rank: Rank):
			self.key = key
			self.val = val
			self.rank = rank
			self.left = None
			self.right = None
