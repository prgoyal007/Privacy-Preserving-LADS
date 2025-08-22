import random
from structures.skiplist import SkipList, Node as SkipNode
from structures.biased_skiplist import BiasedSkipList, Node as BiasedSkipNode
from structures.biased_zipzip import ZipZipTree, Node as ZipZipNode

def generate_dataset(n):
    # keys from 0..n-1
    keys = list(range(n))
    
    # generate random positive numbers
    raw_freqs = [random.randint(1, n) for _ in range(n)]
    
    # normalize so they sum to 1
    total = sum(raw_freqs)
    freqs = [f/total for f in raw_freqs]

    # store in dict {key: frequency}
    freq_dict = {k: f for k, f in zip(keys, freqs)}
    
    return keys, freq_dict


if __name__ == "__main__":
    n = 10  # size of dataset
    keys, freq_dict = generate_dataset(n)
    
    print("Keys:", keys)
    print("Frequencies:", freq_dict)
    print("Sum of frequencies:", sum(freq_dict.values()))
    
    bsl = BiasedSkipList(frequencies=freq_dict, capacity=n)
    
    for key in keys:
        bsl.insert(key)
    
    bsl.display()  # Display the skiplist structure
