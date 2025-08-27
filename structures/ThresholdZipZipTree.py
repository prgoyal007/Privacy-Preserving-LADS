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

class Thresholded_ZipZipTree:
	def __init__(self, capacity: int):
		self.capacity = capacity
		self.root = None
		self.size = 0	# number of nodes in the tree

	def get_random_rank(self, freq: float = None) -> Rank:
		freq = max(freq/2, 1/(2 * self.capacity))
		#freq = max(freq, 1/(self.capacity)) # slightly more optimized?
		
		# Geometric Distribution: # of failures before the first success
		geo_rank = 0
		while random.random() < 0.5:
			geo_rank += 1
		
		if freq is not None:
			# Change geometric rank based on frequency
			geo_rank += math.floor(math.log2(freq * self.capacity))

		# Uniform Distribution: random int from 0 to log(capacity)^3 - 1
		uniform_max = int(math.log(self.capacity, 2) ** 3) - 1
		uniform_rank = random.randint(0, max(uniform_max, 0))

		return Rank(geo_rank, uniform_rank)

	def insert(self, key: KeyType, val: ValType, freq: float, rank: Rank = None):
		# Start by generating a random rank if not provided
		if rank is None:
			rank = self.get_random_rank(freq)	# generate a random rank in form of a tuple (geometric_rank, uniform_rank)
		
		new_node = Node(key, val, rank)	# create a new node with the given key, value, and rank

		self.root = self.insert_node(self.root, new_node)	# insert the new node into the tree
		self.size += 1	# increment the size of the tree


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

	def remove(self, key: KeyType):
		self.root = self.remove_node(self.root, key)	# remove the node with the given key from the tree
		self.size -= 1	# decrement the size of the tree
	
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

	def get_size(self) -> int:
		return self.size

	def get_height(self) -> int:
		return self.recurse_height(self.root)

	def get_depth(self, key: KeyType):
		# similar to get_height, but only for the path from the root to the node with the given key
		return self.recurse_depth(self.root, key, 0)

	def recurse_height(self, node: Node) -> int:
		if not node:
			return -1
		else:
			return 1 + max(self.recurse_height(node.left), self.recurse_height(node.right))
	
	def recurse_depth(self, node: Node, key: KeyType, current_depth: int) -> int:
		if not node:
			raise KeyError(f"Key {key} not found in the tree.")
		
		if key == node.key:
			return current_depth
		elif key < node.key:
			return self.recurse_depth(node.left, key, current_depth + 1)
		else:
			return self.recurse_depth(node.right, key, current_depth + 1)
	
	def is_higher_rank(self, rank1: Rank, rank2: Rank, k1: KeyType = None, k2: KeyType = None) -> bool:
		# Compare the ranks based on the geometric rank first, then the uniform rank
		if rank1.geometric_rank != rank2.geometric_rank:	# compare geometric ranks first
			return rank1.geometric_rank > rank2.geometric_rank
		if rank1.uniform_rank != rank2.uniform_rank:	# compare uniform ranks next
			return rank1.uniform_rank > rank2.uniform_rank
		return k1 < k2 	# tie-breaker based on the keys if ranks are equal
		
	
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
