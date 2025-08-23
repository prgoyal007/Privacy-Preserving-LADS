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


# Parameters for Standard Zipfian Test
ns = [1000, 2000, 5000]
alphas = [1]                            # Only test α = 1
errors = [0]                            # δ = 0 → perfect predictions
search_size = 100000

__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "StandardZipfianTest"

for n in ns:
    for alpha in alphas:
        for error in errors:  # only 0
            print(f"n: {n}, alpha: {alpha}, Standard Zipfian (δ={error})")

            if __generate_data__:
                key_values, search_elements, search_frequencies, ranks = generate_keys(
                    n, search_size, alpha, __random_order__=False
                )

                # true Zipf frequencies (no adversary corruption)
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

            # --- Run on each DS ---
            print(f"n: {n}, alpha: {alpha}, Running DS...")

            p0 = 0.05
            rsl = DynamicRSL(key_values.copy(), frequencies.copy(), p0=p0, right_comparison=False)
            TestDS(rsl, key_values, search_elements, f"{__path_dir__}/RSL_n{n}_a{alpha}.json")

            rsl = DynamicRSL(key_values.copy(), frequencies.copy(), p0=p0, right_comparison=True)
            TestDS(rsl, key_values, search_elements, f"{__path_dir__}/RSLP_n{n}_a{alpha}.json")

            balanced_tree = BinaryTree(key_values, pessimistic=True)
            TestDS(balanced_tree, key_values, search_elements, f"{__path_dir__}/BalBST_n{n}_a{alpha}.json")

            treap = Treap(key_values, frequencies=frequencies)
            TestDS(treap, key_values, search_elements, f"{__path_dir__}/Treap_n{n}_a{alpha}.json")

            avl = AVLTree(key_values)
            TestDS(avl, key_values, search_elements, f"{__path_dir__}/AVL_n{n}_a{alpha}.json")
