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

from tests.DataGenerator import *
from structures.StaticRSL import *
from structures.BiasedZipZipTree import *
from structures.ThresholdZipZipTree import *
from structures.Treap import *
from structures.AVLTree import *



"""
Run a test on a given data structure using either precomputed costs or actual searches.

The function first attempts to use precomputed costs via ds.get_all_costs() if available and 
true_search is False. Otherwise, it performs per-query searches, using ds.find_with_cost() 
when available, or falling back to ds.search().

Parameters:
- ds : DS instance
    The data structure to test (e.g., StaticRSL, Treap, AVLTree, ZipZipTree).
- ordered_elements : list
    The elements in the data structure.
- search_elements : list
    The elements to search for during the test.
- path_to_save : str
    File path to save the results as JSON.
- true_search : bool (default=False)
    If True, perform real searches on the data structure; if False, use precomputed costs.
- __splay_cost__ : bool (default=False)
    For splayable structures, whether to count splay cost (currently unused for static DS).
- __print__ : bool (default=False)
    If True, print debug output.

Returns:
- list : search costs for each element in search_elements.

Side effects:
- Writes search costs to path_to_save as JSON.
"""
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



"""
Save a Python object as JSON to a file.

Parameters:
- data : object
    Data to serialize (typically a list of search costs or dictionary).
- path_to_save : str
    File path to save the JSON.

Side effects:
- Creates or overwrites the specified file with JSON-serialized data.
"""
def write_data(data, path_to_save):
    with open(path_to_save, 'w') as writer:
        json.dump(data, writer)



"""
Read a JSON file and return its contents as a Python object.

Parameters:
- path : str
    Path to the JSON file.

Returns:
- object : the deserialized JSON object (list or dict)

Side effects:
- None
"""
def read_data(path):
    with open(path) as reader:
        return json.load(reader)

# Parameters for Random Zipfian Test
ns = [1000, 2000 ,5000]
alphas = [1, 1.25, 1.5, 2, 3]
errors = [0, 0.01,  0.45, 0.9]
search_size = 100000


__generate_data__ = True
__test_samples__ = True

trials = 10
__path_dir__ = "results/RandomZipfianTest"



"""
Run the Random Zipfian Parameter Test over multiple data structures.

Procedure:
- For each combination of n (size), alpha (Zipfian parameter), and error (adversarial perturbation):
    1. Generate or read key values, search elements, and frequencies.
    2. Apply adversarial Zipfian adjustments to frequencies.
    3. Build each data structure:
        - Static RSL
        - Biased ZipZip Tree
        - Threshold ZipZip Tree
        - Treap
        - AVL Tree
    4. Run TestDS for each structure and save results as JSON.

Side effects:
- Generates JSON files in the results/RandomZipfianTest directory.
- Prints progress and status messages to the console.
"""
for n in ns:
    for alpha in alphas:
        for idx, error in enumerate(errors):
            print(f"n: {n}, alpha: {alpha}, Random Zipfian (δ={error})")

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
            
            print(f"\nStatic RSL size: {static_rsl.get_size()} nodes")
            print(f"Biased ZipZip Tree size: {bzzt.get_size()} nodes")
            print(f"Threshold ZipZip Tree size: {tzzt.get_size()} nodes")
            print(f"Treap size: {treap.get_size()} nodes")
            print(f"AVL Tree size: {avl.get_size()} nodes\n")