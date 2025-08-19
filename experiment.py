import random
from dataset import generate_dataset
from building import build_structures

def generate_queries(keys, freqs, num_queries=10000):
    # keys chosen proportional to their frequency
    # Ex. 
    # keys = ["A", "B", "C"]
    # freqs = {"A": 0.6, "B": 0.3, "C": 0.1}
    # generates A 60% of the time, B 30%, C 10% []
    # ["A", "A", "B", "A", "C", "A", "B", "A", "A", "B"]
    queries = random.choices(
        population=keys,
        weights=[freqs[k] for k in keys],
        k=num_queries
    )
    return queries


def run_experiment(n=100, num_queries=10000):
    sl, bsl, zzt, keys, freqs = build_structures(n)
    queries = generate_queries(keys, freqs, num_queries)

    results = {}

    # SkipList
    total = 0
    for q in queries:
        _, cost = sl.search(q)
        total += cost
    results["SkipList"] = (total, total/num_queries)

    # Biased SkipList
    total = 0
    for q in queries:
        _, cost = bsl.search(q)
        total += cost
    results["BiasedSkipList"] = (total, total/num_queries)

    # ZipZipTree
    total = 0
    for q in queries:
        val, cost = zzt.find_with_cost(q)   # needs a find_with_cost method
        total += cost
    results["ZipZipTree"] = (total, total/num_queries)

    return results


if __name__ == "__main__":
    results = run_experiment(n=10, num_queries=50)
    for name, (total, avg) in results.items():
        print(f"{name}: total cost={total}, avg cost={avg:.3f}")
