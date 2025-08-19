from structures.skiplist import SkipList
from structures.biased_skiplist import BiasedSkipList
from structures.biased_zipzip import ZipZipTree
from dataset import generate_dataset

def build_structures(n):
    # Generate dataset
    keys, freqs = generate_dataset(n)

    # 1. Standard SkipList
    sl = SkipList()
    for k in keys:
        sl.insert(k)

    # 2. Biased SkipList
    bsl = BiasedSkipList(frequencies=freqs, capacity=n)
    for k in keys:
        bsl.insert(k)

    # 3. ZipZipTree
    zzt = ZipZipTree(capacity=n)
    for k in keys:
        # ZipZipTree insert takes value and rank — use k as val for now
        zzt.insert(k, k)

    return sl, bsl, zzt, keys, freqs
