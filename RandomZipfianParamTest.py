"""
Purpose: 
- Generate test sequences where requests (keys/queries) follow a Zipfian distribution with randomly chosen parameters.

How it works:
- Picks a random skew parameter (the α in Zipf's law).
- Uses that α to generate request frequencies (some items are very frequent, others rare).
- Each test run may look different because the skew and distribution vary randomly.

Use case:
- Simulates real-world uncertainty where workloads may not always follow the same skew.
- Good for testing average-case robustness.

Summary: 
- Workload parameters chosen randomly (average-case realism).
- Does the structure handle typical workloads?
"""

import json

from DataGenerator import *
from structures.DynamicRSL import *
from structures.StaticRSL import *
from structures.BiasedZipZipTree import *
from structures.ThresholdZipZipTree import *
from structures.Treap import *
from structures.AVLTree import *

def TestDS(ds, ordered_elements, search_elements, path_to_save, true_search=False, __splay_cost__=False,
           __print__=False):
    costs = []
    if not true_search:
        all_costs_optimistic = ds.get_all_costs(ordered_elements)
        for key in search_elements:
            costs.append(all_costs_optimistic[key])
    else:
        for key in search_elements:
            n, c = ds.search(key, __splay_cost__=__splay_cost__)
            costs.append(c)

    write_data(costs, path_to_save)
    return costs


def write_data(data, path_to_save):
    with open(path_to_save, 'w') as writer:
        json.dump(data, writer)


def read_data(path):
    with open(path) as reader:
        return json.load(reader)

# Parameters for Robust Zipfian Test + Randomness
ns = [1000, 2000 ,5000]
alphas = [1, 1.25, 1.5, 2, 3]
errors = [0, 0.01,  0.45, 0.9]
search_size = 100000


__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "RandomZipfianTest"
for n in ns:
    for alpha in alphas:
        for idx, error in enumerate(errors):
            print("n: {2}, alpha: {1}, Evaluating error: {0}".format(error, alpha, n))

            if __generate_data__:
                if error == 0:
                    key_values, search_elements, search_frequencies, ranks = generate_keys(n, search_size, alpha,
                                                                                           __random_order__=True)
                else:
                    data = read_data("{2}/data_n{3}_e{0}_a{1}.json".format(0, alpha, __path_dir__, n))
                    key_values = data['keys']
                    search_elements = data['search']
                    ranks = data['ranks']
                    search_frequencies = data['search_freq']

                frequencies = zipfi_adversary(n, ranks, error, alpha)

                data = {}
                data['keys'] = key_values
                data['search'] = search_elements
                data['freq'] = frequencies
                data['search_freq'] = search_frequencies
                data['ranks'] = ranks
                write_data(data, "{2}/data_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))




            else:
                data = read_data("{2}/data_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))
                key_values = data['keys']
                search_elements = data['search']
                frequencies = data['freq']
                search_frequencies = data['search_freq']

            # impacted by frequencies
            print("n: {1}, alpha: {0}, Making RSL...".format(alpha, n))
            p0 = min(0.9, max(search_frequencies) * 0.9)
            rsl = DynamicRSL(key_values.copy(), frequencies.copy(), p0=p0, right_comparison=False)

            TestDS(rsl, key_values, search_elements,
                   "{2}/RSL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            print("n: {1}, alpha: {0}, Making RSL...".format(alpha, n))
            p0 = min(0.9, max(search_frequencies) * 0.9)
            rsl = DynamicRSL(key_values.copy(), frequencies.copy(), p0=p0, right_comparison=True)

            TestDS(rsl, key_values, search_elements,
                   "{2}/RSLP_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            print("n: {1}, alpha: {0}, Making balanced BST...".format(alpha, n))
            balanced_tree = BinaryTree(key_values, pessimistic=True)

            TestDS(balanced_tree, key_values, search_elements,
                   "{2}/BalBST_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))


            # impacted by frequencies
            print("n: {1}, alpha: {0}, Making Treap...".format(alpha, n))
            treap = Treap(key_values, frequencies=frequencies)

            TestDS(treap, key_values, search_elements,
                   "{2}/Treap_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))


            print("n: {1}, alpha: {0}, Making AVL Tree...".format(alpha, n))
            avl = AVLTree(key_values)

            TestDS(avl, key_values, search_elements,
                   "{2}/AVL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))