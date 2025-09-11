import json

from tests.DataGenerator import *
from structures.StaticRSL import *
from structures.BiasedZipZipTree import *
from structures.ThresholdZipZipTree import *
from structures.PairedZipZipTree import *
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
ns = [100, 500, 1000, 2000, 5000, 10000]
alphas = [2]
errors = [0]
search_size = 100000

__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "results/SizeTest"

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


            # Only Static RobustSL            
            print(f"n: {n}, alpha: {alpha}, Making RobustSL...")
            static_rsl = StaticRSL(key_values.copy(), frequencies.copy(), right_comparison=True)
            TestDS(static_rsl, key_values, search_elements,
                   "{2}/RobustSL_n{3}_e{0}_a{1}.json".format(int(error * 100), alpha, __path_dir__, n))
            
            print(f"\nRobustSL size: {static_rsl.get_size()} nodes")