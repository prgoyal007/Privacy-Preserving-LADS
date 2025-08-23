"""
Purpose:
- Generate test sequences where requests follow a Zipfian distribution with *fixed, standard parameters*.
- Serves as the baseline workload model since Zipfian distributions are common in real-world access patterns (web traffic, database queries, caching).

How it works:
- Chooses a dataset size (N) and a Zipf parameter (α).
- Requests are drawn directly from the Zipf(α) distribution without adversarial tuning.
- Typical α values:
    - 0.8 ≤ α ≤ 1.2 → realistic skew (many moderately hot items, some very hot items).
- Produces “natural” heavy-tailed frequency patterns.

Use case:
- Benchmarking baseline performance of data structures under realistic non-uniform workloads.
- Comparison point against adversarial or randomized variants.

Summary:
- Standard Zipf distribution with fixed parameters.
- Models everyday workloads with skewed popularity.
"""


import json

from DataGenerator import *
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


# Parameters for Standard Zipfian Test
ns = [1000, 2000, 5000]
alphas = [1]                            # Only test α = 1
errors = [0]                            # δ = 0 → perfect predictions
search_size = 100000

__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "results/StandardZipfianTest"

for n in ns:
    for alpha in alphas:
        for error in errors:                                                                    # only 0
            print(f"n: {n}, alpha: {alpha}, Standard Zipfian (δ={error})")

            if __generate_data__:
                key_values, search_elements, search_frequencies, ranks = generate_keys(
                    n, search_size, alpha, __random_order__=False
                )

                # true Zipfian frequencies (no adversary corruption)
                frequencies = zipfi_adversary(n, ranks, error, alpha)

                data = {
                    "keys": key_values,
                    "search": search_elements,
                    "freq": frequencies,
                    "search_freq": search_frequencies,
                    "ranks": ranks,
                }
                write_data(data, f"{__path_dir__}/data_n{n}_a{alpha}.json")

            else:
                data = read_data(f"{__path_dir__}/data_n{n}_a{alpha}.json")
                key_values = data["keys"]
                search_elements = data["search"]
                frequencies = data["freq"]
                search_frequencies = data["search_freq"]

            # Static RSL
            print(f"n: {n}, alpha: {alpha}, Making Static RSL...")
            static_rsl = StaticRSL(key_values.copy(), frequencies.copy())
            TestDS(static_rsl, key_values, search_elements, f"{__path_dir__}/StaticRSL_n{n}_a{alpha}.json")

            # Biased ZipZip Tree
            print(f"n: {n}, alpha: {alpha}, Making Biased ZipZip Tree...")
            bzzt = ZipZipTree(n)
            for key, freq in zip(key_values, frequencies):
                bzzt.insert(key, key, freq)
            TestDS(bzzt, key_values, search_elements, f"{__path_dir__}/BiasedZipZipTree_n{n}_a{alpha}.json")

            # Threshold Biased ZipZip Tree
            print(f"n: {n}, alpha: {alpha}, Making Threshold ZipZip Tree...")
            tzzt = Thresholded_ZipZipTree(n)
            for key, freq in zip(key_values, frequencies):
                tzzt.insert(key, key, freq)
            TestDS(tzzt, key_values, search_elements, f"{__path_dir__}/ThresholdZipZipTree_n{n}_a{alpha}.json")

            # Treap
            print(f"n: {n}, alpha: {alpha}, Making Treap...")
            treap = Treap(key_values, frequencies=frequencies)
            TestDS(treap, key_values, search_elements, f"{__path_dir__}/Treap_n{n}_a{alpha}.json")

            # AVL Tree
            print(f"n: {n}, alpha: {alpha}, Making AVL Tree...")
            avl = AVLTree(key_values)
            TestDS(avl, key_values, search_elements, f"{__path_dir__}/AVL_n{n}_a{alpha}.json")
