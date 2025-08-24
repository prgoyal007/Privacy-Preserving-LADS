class DS:

    """
    Search for a given key in the data structure.

    Parameters:
    - key_Value : the key to search for
    - __splay_cost__ : optional flag for splay tree cost tracking (default False)

    Returns:
    - tuple (key_Value, cost) where
        - key_Value = the searched key
        - cost = number of steps/comparisons (currently always 0 in base case)
    """
    def search(self, key_Value, __splay_cost__=False):
        cost = 0
        return key_Value, cost

    """
    Get the search costs for a list of keys.

    Parameters:
    - keys : iterable of keys to search

    Return:
    - dict mapping each key to its search cost
    """
    def get_all_costs(self, keys):
        costs = {}
        for key in keys:
            try:
                _, cost = self.search(key)      # Perform search (subclass defines actual behavior)
                costs[key] = cost               # Record the cost for this key
            except Exception as e:
                print("Couldn't find element with key: {0}".format(key))

        return costs