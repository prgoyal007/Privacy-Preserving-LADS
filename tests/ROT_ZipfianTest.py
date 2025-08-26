import json

from tests.DataGenerator import *
from structures.StaticRSL import *
from structures.BiasedZipZipTree import *
from structures.ThresholdZipZipTree import *
from structures.LTreap import *
from structures.CTreap import *
from structures.AVLTree import *

def TestDS(ds, ordered_elements, search_elements, path_to_save, true_search=False, __splay_cost__=False,
           __print__=False):
    costs = []

    if not true_search and hasattr(ds, "get_all_costs"):
        # Use precomputed costs if available
        all_costs_optimistic = ds.get_all_costs(ordered_elements)
        for key in search_elements:
            costs.append(all_costs_optimistic[key])
    else:
        # Fall back to running per-query searches
        for key in search_elements:
            if hasattr(ds, "find_with_cost"):
                _, c = ds.find_with_cost(key)
            else:
                _, c = ds.search(key, __splay_cost__=__splay_cost__)
            costs.append(c)
    size = ds.get_size() if hasattr(ds, "get_size") else None
    results = {
        "costs": costs,
        "size": size
    }

    write_data(results, path_to_save)
    return costs

def write_data(data, path_to_save):
    with open(path_to_save, 'w') as writer:
        json.dump(data, writer)


def read_data(path):
    with open(path) as reader:
        return json.load(reader)

# Parameters for Zipfian Tests where Random Order = True
ns = [100, 500, 1000, 2000]
alphas = [2]
errors = [0]
search_size = 100000

__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "results/ROTZipfianTest"

for n in ns:
    for alpha in alphas:
        for idx, error in enumerate(errors):
            print(f"n: {n}, alpha: {alpha}, Randomized Zipfian (δ={error})")

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


            # Static RobustSL            
            print(f"n: {n}, alpha: {alpha}, Making RobustSL...")
            static_rsl = StaticRSL(key_values.copy(), frequencies.copy(), right_comparison=True)
            TestDS(static_rsl, key_values, search_elements,
                   "{2}/RobustSL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Threshold Biased ZipZip Tree
            print(f"n: {n}, alpha: {alpha}, Making Threshold ZipZip Tree...")
            tzzt = Thresholded_ZipZipTree(n)
            for key, freq in zip(key_values, frequencies):
                tzzt.insert(key, key, freq)
            TestDS(tzzt, key_values, search_elements,
                   "{2}/ThresholdZipZipTree_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Biased ZipZip Tree
            print(f"n: {n}, alpha: {alpha}, Making Biased ZipZip Tree...")
            bzzt = ZipZipTree(n)
            for key, freq in zip(key_values, frequencies):
                bzzt.insert(key, key, freq)
            TestDS(bzzt, key_values, search_elements,
                   "{2}/BiasedZipZipTree_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Chen's Treap
            print(f"n: {n}, alpha: {alpha}, Making CTreap...")
            ctreap = Treap(key_values, frequencies=frequencies, log_priority=True)
            TestDS(ctreap, key_values, search_elements,
                   "{2}/CTreap_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # Lin's Treap
            print(f"n: {n}, alpha: {alpha}, Making LTreap...")
            ltreap = Treap(key_values, frequencies=frequencies, log_priority=False)
            TestDS(ltreap, key_values, search_elements,
                   "{2}/LTreap_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))

            # AVL Tree
            print(f"n: {n}, alpha: {alpha}, Making AVL Tree...")
            avl = AVLTree(key_values)
            TestDS(avl, key_values, search_elements,
                   "{2}/AVL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))
            
            print(f"\nRobustSL size: {static_rsl.get_size()} nodes")
            print(f"Threshold ZipZip Tree size: {tzzt.get_size()} nodes")
            print(f"Biased ZipZip Tree size: {bzzt.get_size()} nodes")
            print(f"Chen's Treap size: {ctreap.get_size()} nodes")
            print(f"Lin's Treap size: {ltreap.get_size()} nodes")
            print(f"AVL Tree size: {avl.get_size()} nodes\n")