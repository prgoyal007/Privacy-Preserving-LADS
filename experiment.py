import random
from dataset import generate_dataset
from building import build_structures

def generate_queries(keys, freq_dict, num_queries=10000):
    # keys chosen proportional to their frequency
    # Ex. 
    # keys = ["A", "B", "C"]
    # freq_dict = {"A": 0.6, "B": 0.3, "C": 0.1}
    # generates A 60% of the time, B 30%, C 10% []
    # ["A", "A", "B", "A", "C", "A", "B", "A", "A", "B"]
    queries = random.choices(
        population=keys,
        weights=[freq_dict[k] for k in keys],
        k=num_queries
    )
    return queries

def run_experiment(n=100, num_queries=10000):
    # Build data structures
    zzt, thresholded_zzt, keys, freq_dict = build_structures(n)

    # Generate queries proportional to frequencies
    queries = generate_queries(keys, freq_dict, num_queries)

    results = {}

    # Test ZipZipTree
    total = 0
    for q in queries:
        val, cost = zzt.find_with_cost(q)                                   # needs a find_with_cost method
        total += cost
    results["ZipZipTree"] = (total, total / num_queries)

    # Test Thresholded ZipZipTree
    total = 0
    for q in queries:
        val, cost = thresholded_zzt.find_with_cost(q)
        total += cost
    results["Thresholded ZipZipTree"] = (total, total / num_queries)

    return results

if __name__ == "__main__":
    n = 10
    num_queries = 100
    print(f"\n=== Running Experiment: n={n}, num_queries={num_queries} ===\n")
    results = run_experiment(n=n, num_queries=num_queries)

    for name, (total, avg) in results.items():
        print(f"{name}: Total cost: {total}, Average cost per query: {avg:.3f}")
    
    print("\n=== Experiment Complete ===\n")
