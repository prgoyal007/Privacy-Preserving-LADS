class DS:

    def search(self, key_Value, __splay_cost__=False):
        cost = 0
        return key_Value, cost

    def get_all_costs(self, keys):
        costs = {}
        for key in keys:
            try:
                _, cost = self.search(key)
                costs[key] = cost
            except Exception as e:
                print("Couldn't find element with key: {0}".format(key))

        return costs