"""
Purpose: 
- Generate test sequences where requests follow a Zipfian distribution but with adversarially chosen parameters (worst-case scenarios).

How it works:
- Instead of sampling α randomly, it picks α (and sometimes dataset size) in a way that is maximally stressful to the data structure.
- For example:
    - Extreme skew (α → 2): almost all queries target a single element.
    - Near-uniform skew (α → 0): no caching or optimization works well.
- This can reveal pathological performance issues that random tests might miss.

Use case:
- Stress-tests against worst-case workloads.
- Helps analyze history independence/privacy leaks under “attack-style” access patterns.

Summary:
- Workload parameters chosen adversarially (worst-case stress).
- Does the structure break under skew extremes?
"""

import json

from tests.DataGenerator import *
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

# Parameters for Adversarial Zipfian Test
ns = [1000, 2000 ,5000]
alphas = [1, 1.25, 1.5, 2, 3]
errors = [0, 0.01,  0.45, 0.9]
search_size = 100000

__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "results/AdvZipfianTest"

for n in ns:
    for alpha in alphas:
        for idx, error in enumerate(errors):
            print("n: {2}, alpha: {1}, Evaluating error: {0}".format(error, alpha, n))

            if __generate_data__:
                if error == 0:
                    key_values, search_elements, search_frequencies, ranks = generate_keys(n, search_size, alpha,
                                                                                           __random_order__=False)
                else:
                    data = read_data("{2}/data_n{3}_e{0}_a{1}.json".format(0, alpha, __path_dir__, n))
                    key_values = data['keys']
                    search_elements = data['search']
                    ranks = data['ranks']
                    search_frequencies = data['search_freq']

                frequencies = zipfi_adversary(n, ranks, error, alpha)

                data = {
                    'keys': key_values,
                    'search': search_elements,
                    'freq': frequencies,
                    'search_freq': search_frequencies,
                    'ranks': ranks
                }
                write_data(data, "{2}/data_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            else:
                data = read_data("{2}/data_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))
                key_values = data['keys']
                search_elements = data['search']
                frequencies = data['freq']
                search_frequencies = data['search_freq']


            # Static RSL            
            print(f"n: {n}, alpha: {alpha}, Making Static RSL...")
            static_rsl = StaticRSL(key_values.copy(), frequencies.copy())
            TestDS(static_rsl, key_values, search_elements,
                   "{2}/StaticRSL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Biased ZipZip Tree
            print(f"n: {n}, alpha: {alpha}, Making Biased ZipZip Tree...")
            bzzt = ZipZipTree(n)
            for key, freq in zip(key_values, frequencies):
                bzzt.insert(key, key, freq)
            TestDS(bzzt, key_values, search_elements,
                   "{2}/BiasedZipZipTree_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Threshold Biased ZipZip Tree
            print(f"n: {n}, alpha: {alpha}, Making Threshold ZipZip Tree...")
            tzzt = Thresholded_ZipZipTree(n)
            for key, freq in zip(key_values, frequencies):
                tzzt.insert(key, key, freq)
            TestDS(tzzt, key_values, search_elements,
                   "{2}/ThresholdZipZipTree_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Treap
            print(f"n: {n}, alpha: {alpha}, Making Treap...")
            treap = Treap(key_values, frequencies=frequencies)
            TestDS(treap, key_values, search_elements,
                   "{2}/Treap_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # AVL Tree
            print(f"n: {n}, alpha: {alpha}, Making AVL Tree...")
            avl = AVLTree(key_values)
            TestDS(avl, key_values, search_elements,
                   "{2}/AVL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))
            