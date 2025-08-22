from structures.BiasedZipZipTree import ZipZipTree
from structures.ThresholdZipZipTree import Thresholded_ZipZipTree
from dataset import generate_dataset

def build_structures(n):
    # Generate dataset
    keys, freq_dict = generate_dataset(n)

    # 3. ZipZipTree
    zzt = ZipZipTree(capacity=n)
    for k in keys:
        # Insert k as both key and value, rank is generated on its own
        zzt.insert(k, k, freq_dict[k])
    
    thresholded_zzt = Thresholded_ZipZipTree(capacity=n)
    for k in keys:
        # Insert k as both key and value, rank is generated on its own
        thresholded_zzt.insert(k, k, freq_dict[k])

    return zzt, thresholded_zzt, keys, freq_dict
