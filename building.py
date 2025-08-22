from structures.skiplist import SkipList
from structures.biased_skiplist import BiasedSkipList
from structures.biased_zipzip import ZipZipTree
from structures.zipzip_thresholded import Thresholded_ZipZipTree
from dataset import generate_dataset

def build_structures(n):
    # Generate dataset
    keys, freq_dict = generate_dataset(n)

    # 1. Standard SkipList
    sl = SkipList()
    for k in keys:
        sl.insert(k)

    # 2. Biased SkipList
    bsl = BiasedSkipList(frequencies=freq_dict, capacity=n)
    for k in keys:
        bsl.insert(k)

    # 3. ZipZipTree
    zzt = ZipZipTree(capacity=n)
    for k in keys:
        # Insert k as both key and value, rank is generated on its own
        zzt.insert(k, k, freq_dict[k])
    
    thresholded_zzt = Thresholded_ZipZipTree(capacity=n)
    for k in keys:
        # Insert k as both key and value, rank is generated on its own
        thresholded_zzt.insert(k, k, freq_dict[k])

    return sl, bsl, zzt, thresholded_zzt, keys, freq_dict
