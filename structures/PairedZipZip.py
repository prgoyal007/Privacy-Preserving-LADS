from structures.BiasedZipZipTree import *

gamma = 1

class PairedZipZipTree:

	"""
	Initialize a PairedZipZipTree.

	Parameters:
	- capacity : Maximum number of nodes expected (affects rank scaling)
	"""
	def __init__(self, capacity: int):
		self.biased_zip = BiasedZipZipTree(capacity)
		self.standard_zip = BiasedZipZipTree(capacity)
		self.limit_walk = gamma
	

	"""
	Insert a new key-value pair into the ZipZipTree, optionally using a given rank.

	Parameters:
	- key : KeyType
		The key of the new node to insert.
	- val : ValType
		The value associated with the key.
	- freq : float
		Optional frequency to influence rank generation.
	- rank : Rank (optional)
		Precomputed rank to use instead of generating a random one.

	Side effects:
	- Updates self.root to reflect the inserted node.
	- Increments self.size by 1.
	"""
	def insert(self, key: KeyType, val: ValType, freq: float, rank: Rank = None):
		self.biased_zip.insert(key, val, freq, rank)
		self.standard_zip.insert(key, val, None, rank)


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
		self.biased_zip.remove(key)
		self.standard_zip.remove(key)

	
	"""
	Find the value associated with a key.

	Parameters:
	- key : KeyType
		The key to search for.

	Returns:
	- Value associated with the key if found, else None.
	"""
	def find(self, key: KeyType):
		return self.standard_zip.find()
	

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
		n = self.biased_zip.get_size()
		cost1, cost2 = 0
		found, cost1 = self.biased_zip.find_with_cost(key, self.limit_walk * math.floor(math.log2(n)))

		if found is False:
			true_found, cost2 = self.standard_zip.find_with_cost(key)
		
		total_cost = cost1 + cost2

		return true_found, total_cost
	

	"""
	Return the current number of nodes in the ZipZipTree.

	Returns:
	- int : number of nodes in the tree (self.size)
	"""
	def get_size(self) -> int:
		return self.biased_zip.get_size() + self.standard_zip.get_size()
	
	
	"""
	Compute the height of the ZipZipTree.

	Returns:
	- int : height of the tree (max depth from root to any leaf)	
	"""
	def get_height(self) -> int:
		return self.biased_zip.get_height()
	
