import glob

import numpy as np

from nltk.tokenize import word_tokenize


def generate_keys(n, search_size, alpha, __random_order__=False):
    key_values = list(range(n))

    ranks = [i + 1 for i in range(n)]
    if __random_order__:
        np.random.shuffle(ranks)
    search_frequencies = [1 / (ranks[i] ** alpha) for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    return key_values, [int(s) for s in search_elements], [float(f) for f in search_frequencies], ranks


def generate_dynamic_keys(n, search_size, alpha, __random_order__=False, reversed=False, insert_ratio=0.2):
    key_values = list(range(n))

    ranks = [i + 1 for i in range(n)]
    if __random_order__:
        np.random.shuffle(ranks)
    if reversed:
        ranks.reverse()
    search_frequencies = [1 / (ranks[i] ** alpha) for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)

    # if __random_order__:
    #     error_ranks = [max(1, ranks[i] * np.random.normal(0, error))  for i in range(n)]
    # else:
    #     error_ranks = [ranks[i] * (1 - error) + error * (n - ranks[i] + 1) for i in range(n)]

    data = {}
    for i in range(len(key_values)):
        data[key_values[i]] = search_frequencies[i]

    live_elements = []
    dead_elements = []
    for v in key_values:
        if np.random.random() < insert_ratio:
            dead_elements.append(v)
        else:
            live_elements.append(v)
    initial_elements = live_elements.copy()

    queries = []
    for i in range(search_size):
        query = {}
        rnd = np.random.random()
        if rnd < insert_ratio:
            # insert
            if len(dead_elements) == 0:
                continue
            e = np.random.choice(dead_elements, 1)[0]
            e = int(e)
            query['type'] = "insert"
            query['key'] = e
            # query['freq'] = predicted_freqs[e]

            dead_elements.remove(e)
            live_elements.append(e)
            queries.append(query)
        else:
            if len(live_elements) == 0:
                continue
            e = np.random.choice(key_values, 1, p=search_frequencies)[0]
            if e not in live_elements:
                query['type'] = "insert"
                query['key'] = int(e)
                # query['freq'] = predicted_freqs[e]
                live_elements.append(e)
                dead_elements.remove(e)
                queries.append(query)

            query = {}
            query['type'] = 'search'
            query['key'] = int(e)
            queries.append(query)

    return key_values, initial_elements, queries, [float(f) for f in search_frequencies], ranks


def zipfi_dynamic_fixed(n, key_values, initial_elements, ranks, queries, error, alpha):
    error_ranks = [ranks[i] * (1 - error) + error * (n - ranks[i] + 1) for i in range(n)]

    frequencies = [1 / (r ** alpha) for r in error_ranks]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    predicted_freqs = {}
    for i in range(len(key_values)):
        predicted_freqs[key_values[i]] = frequencies[i]

    initial_frequencies = [predicted_freqs[e] for e in initial_elements]

    for query in queries:
        e = int(query['key'])
        query['freq'] = predicted_freqs[e]

    return queries, initial_frequencies


def zipfi_adversary_dynamic(n, search_size, error, alpha, __random_order__=False, insert_ratio=0.2):
    key_values = list(range(n))

    ranks = [i + 1 for i in range(n)]
    if __random_order__:
        np.random.shuffle(ranks)
    search_frequencies = [1 / (ranks[i] ** alpha) for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)

    error_ranks = [ranks[i] * (1 - error) + error * (n - ranks[i] + 1) for i in range(n)]
    frequencies = [1 / (r ** alpha) for r in error_ranks]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    data = {}
    for i in range(len(key_values)):
        data[key_values[i]] = search_frequencies[i]

    predicted_freqs = {}
    for i in range(len(key_values)):
        predicted_freqs[key_values[i]] = frequencies[i]
    live_elements = []
    dead_elements = []
    for v in key_values:
        if np.random.random() < insert_ratio:
            dead_elements.append(v)
        else:
            live_elements.append(v)
    initial_elements = live_elements.copy()
    initial_frequencies = [predicted_freqs[e] for e in initial_elements]
    queries = []
    for i in range(search_size):
        query = {}
        rnd = np.random.random()
        if rnd < insert_ratio * 3 / 4:
            # insert
            if len(dead_elements) == 0:
                continue
            e = np.random.choice(dead_elements, 1)[0]
            e = int(e)
            query['type'] = "insert"
            query['key'] = e
            query['freq'] = predicted_freqs[e]

            dead_elements.remove(e)
            live_elements.append(e)
            queries.append(query)
        elif rnd < insert_ratio:
            # delete
            if len(live_elements) == 0:
                continue
            e = np.random.choice(live_elements, 1)[0]
            e = int(e)
            query['type'] = 'delete'
            query['key'] = e
            live_elements.remove(e)
            dead_elements.append(e)
            queries.append(query)
        else:
            if len(live_elements) == 0:
                continue
            e = np.random.choice(live_elements, 1, p=get_frequencies(live_elements, data))[0]
            e = int(e)
            query['type'] = 'search'
            query['key'] = e
            queries.append(query)

    return initial_elements, queries, [float(f) for f in initial_frequencies], [float(f) for f in
                                                                                search_frequencies]


def random_samples(n, search_size, error):
    key_values = list(range(n))
    frequencies = [np.random.random() for _ in range(n)]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    search_frequencies = [frequencies[i] * (1 - error) + np.random.random() * (error) for i in range(len(frequencies))]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    return key_values, search_elements, frequencies, search_frequencies


def optimistic_worst(n, search_size):
    key_values = list(range(n))
    epsilon = 0.001 / n
    frequencies = [(n - i) * epsilon for i in range(n)]
    frequencies[0] = 1
    frequencies = np.array(frequencies) / np.sum(frequencies)

    search_frequencies = [0 for _ in range(n)]
    search_frequencies[-1] = 1
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    return key_values, search_elements, frequencies, search_frequencies


def pessimistic_worst(n, search_size):
    key_values = list(range(n))
    epsilon = 0.001 / n
    frequencies = [(n - i) * epsilon for i in range(n)]
    frequencies[0] = 1
    frequencies = np.array(frequencies) / np.sum(frequencies)

    search_frequencies = [0 for _ in range(n)]
    search_frequencies[0] = 1
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    return key_values, search_elements, frequencies, search_frequencies


def zipfi_freq(n, search_size, error, alpha):
    key_values = list(range(n))

    search_frequencies = [1 / (i + 1) ** alpha for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    frequencies = [np.random.uniform(v / (1 + error), v * (1 + error)) for v in search_frequencies]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    return key_values, [int(s) for s in search_elements], [float(f) for f in frequencies], [float(f) for f in
                                                                                            search_frequencies]


def zipfi(n, search_size, error, alpha):
    key_values = list(range(n))

    ranks = [i + 1 for i in range(n)]
    np.random.shuffle(ranks)
    search_frequencies = [1 / (ranks[i] ** alpha) for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    error_ranks = [np.random.uniform(ranks[i] / (1 + error), ranks[i] * (1 + error)) for i in range(n)]
    error_ranks = [max(0.01, v) for v in error_ranks]
    frequencies = [1 / (r ** alpha) for r in error_ranks]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    return key_values, [int(s) for s in search_elements], [float(f) for f in frequencies], [float(f) for f in
                                                                                            search_frequencies]


def zipfi_limited(n, search_size, error, alpha):
    key_values = list(range(n))

    ranks = [i + 1 for i in range(n)]
    np.random.shuffle(ranks)
    search_frequencies = [1 / (ranks[i] ** alpha) for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    error_ranks = [np.random.uniform(max(ranks[i] / (1 + error), 1), min(ranks[i] * (1 + error), n)) for i in range(n)]
    frequencies = [1 / (r ** alpha) for r in error_ranks]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    return key_values, [int(s) for s in search_elements], [float(f) for f in frequencies], [float(f) for f in
                                                                                            search_frequencies]


def zipfi_adversary(n, ranks, error, alpha):
    # key_values = list(range(n))
    #
    # ranks = [i + 1 for i in range(n)]
    # if __random_order__:
    #     np.random.shuffle(ranks)
    # search_frequencies = [1 / (ranks[i] ** alpha) for i in range(n)]
    # search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    # search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    error_ranks = [ranks[i] * (1 - error) + error * (n - ranks[i] + 1) for i in range(n)]
    frequencies = [1 / (r ** alpha) for r in error_ranks]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    return [float(f) for f in frequencies]


def zipfi_lin(n, search_size, error, alpha):
    key_values = list(range(n))

    ranks = [i + 1 for i in range(n)]
    np.random.shuffle(ranks)
    frequencies = [1 / ranks[i] ** alpha for i in range(n)]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    search_frequencies = [frequencies[i] * (1 + 2 * (np.random.random() - 0.5) * error) for i in
                          range(len(frequencies))]
    search_frequencies = [max(v, 1 / (100 * n)) for v in search_frequencies]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    return key_values, [int(s) for s in search_elements], [float(f) for f in frequencies], [float(f) for f in
                                                                                            search_frequencies]


def gaussian_density(x, mu, var):
    return (np.pi * var) * np.exp(-0.5 * ((x - mu) / var) ** 2)


def Gaussian(n, search_size, error, sigma):
    key_values = list(range(n))
    ranks = [i for i in range(n)]
    # np.random.shuffle(ranks)
    search_frequencies = [gaussian_density(ranks[i], 0, sigma) for i in range(n)]
    search_frequencies = np.array(search_frequencies) / np.sum(search_frequencies)
    search_elements = np.random.choice(key_values, search_size, p=search_frequencies)

    error_ranks = [np.random.uniform(max(ranks[i] / (1 + error), 0), min(ranks[i] * (1 + error), n - 1)) for i in
                   range(n)]
    frequencies = [gaussian_density(r, 0, sigma) for r in error_ranks]
    frequencies = np.array(frequencies) / np.sum(frequencies)

    return key_values, [int(s) for s in search_elements], [float(f) for f in frequencies], [float(f) for f in
                                                                                            search_frequencies]


def get_frequencies(elements, data):
    frequencies = []
    for e in elements:
        frequencies.append(data[e])
    frequencies = np.array(frequencies) / np.sum(frequencies)
    return frequencies


def hetro_dynamic_fixed(n, search_size, h, error, ordered=False, reversed=False, insert_ratio=0.2):
    key_values = list(range(n))

    p_min = 1e-3
    p_max = 0.2
    delta = (p_max - p_min) / (n - 1)
    search_frequencies = []
    low_f = int(n * (1 - insert_ratio / 10))
    print(low_f)
    for i, e in enumerate(key_values):
        if i < low_f:
            search_frequencies.append((1 - h) / n + h * (p_min + delta * i))
        else:
            search_frequencies.append((1 - h) / n + h * (n * p_max))

    sum_f = 0
    for f in search_frequencies:
        sum_f += f
    for i in range(len(search_frequencies)):
        search_frequencies[i] /= sum_f

    if reversed:
        search_frequencies.reverse()
    if not ordered:
        np.random.shuffle(search_frequencies)

    predicted_frequencies = []
    for i, e in enumerate(search_frequencies):
        predicted_frequencies.append(search_frequencies[i] * (1 + np.random.normal(0, error)))
    sum_f = 0
    for f in predicted_frequencies:
        sum_f += f
    for i in range(len(predicted_frequencies)):
        predicted_frequencies[i] /= sum_f

    predicted_f = []

    initial_elements = []
    live_elements = []
    dead_elements = []

    for i in range(len(search_frequencies)):
        e = key_values[i]
        if np.random.random() < insert_ratio:
            dead_elements.append(e)
        else:
            initial_elements.append(e)
            live_elements.append(e)
            predicted_f.append(predicted_frequencies[i])
    queries = []
    low_count = search_size
    for i in range(search_size):
        query = {}
        rnd = np.random.random()
        if rnd < insert_ratio:
            # insert
            if len(dead_elements) == 0:
                continue
            e = np.random.choice(dead_elements, 1)[0]
            e = int(e)
            query['type'] = "insert"
            query['key'] = e
            query['freq'] = predicted_frequencies[e]

            dead_elements.remove(e)
            live_elements.append(e)
            queries.append(query)
        else:
            e = np.random.choice(key_values, 1, p=search_frequencies)[0]
            if e not in live_elements:
                query['type'] = "insert"
                query['key'] = e
                query['freq'] = predicted_frequencies[e]
                live_elements.append(e)
                dead_elements.remove(e)
                queries.append(query)

            query = {}
            query['type'] = 'search'
            query['key'] = e
            if e < low_f:
                low_count -= 1
            queries.append(query)
    print("% of low f queries: {0:.1%}".format(low_count / search_size))
    return initial_elements, queries, predicted_f, search_frequencies


def get_bbc_dataset(path_directory, dictionary_size=2000, train_ratio=0.5):
    # folder path
    # path_directory = "dataset\\bbc\\business\\"

    files = path_directory + "*.txt"
    files = glob.glob(files)
    n = len(files)
    n1 = int(train_ratio * n)
    n2 = n - n1
    data = {}
    for i in range(n1):
        file_content = open(files[i]).read()
        tokens = word_tokenize(file_content)
        for token in tokens:
            if token not in data:
                data[token] = 1
            else:
                data[token] += 1
    high_frequent_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:dictionary_size]
    key_values_string = []
    predicted_frequencies = []
    for (key, freq) in high_frequent_items:
        key_values_string.append(key)
        predicted_frequencies.append(freq)
    predicted_frequencies = np.array(predicted_frequencies) / np.sum(predicted_frequencies)

    sorted_key_elements = sorted(key_values_string)
    key_dictionary = {}
    key_values = list(range(len(sorted_key_elements)))
    for i in range(len(sorted_key_elements)):
        key_dictionary[sorted_key_elements[i]] = i

    search_elements = []
    for i in range(n):
        file_content = open(files[i]).read()
        tokens = word_tokenize(file_content)
        for token in tokens:
            if token in key_dictionary:
                search_elements.append(key_dictionary[token])
    np.random.shuffle(search_elements)

    return key_values, search_elements, predicted_frequencies, predicted_frequencies


def get_bbc_dataset_noisy(path_directory, search_size=150000, dictionary_size=2000, train_size=200, scale_factor=1):
    # folder path
    # path_directory = "dataset\\bbc\\business\\"

    files = path_directory + "*.txt"
    files = glob.glob(files)
    n = len(files)
    n1 = train_size
    n2 = n
    data = {}
    search_data = {}

    for i in range(n):
        file_content = open(files[i]).read()
        tokens = word_tokenize(file_content)
        for token in tokens:
            if i < n1:
                if token not in data:
                    data[token] = 1
                else:
                    data[token] += 1
            if token not in search_data:
                search_data[token] = 1
            else:
                search_data[token] += 1
    common_items = {}
    for key in data:
        common_items[key] = search_data[key]
    high_frequent_items = sorted(common_items.items(), key=lambda x: x[1], reverse=True)[:dictionary_size]
    key_values_string = []
    predicted_frequencies = []
    actual_frequencies = []
    for (key, freq) in high_frequent_items:
        key_values_string.append(key)
        predicted_frequencies.append(data[key] ** scale_factor)
        actual_frequencies.append(search_data[key] ** scale_factor)
    predicted_frequencies = np.array(predicted_frequencies) / np.sum(predicted_frequencies)
    actual_frequencies = np.array(actual_frequencies) / np.sum(actual_frequencies)

    sorted_key_elements = sorted(key_values_string)
    key_dictionary = {}

    for i in range(len(sorted_key_elements)):
        key_dictionary[sorted_key_elements[i]] = i

    key_values = []
    for key in key_values_string:
        key_values.append(key_dictionary[key])

    # search_elements = []
    # for i in range(n):
    #     file_content = open(files[i]).read()
    #     tokens = word_tokenize(file_content)
    #     for token in tokens:
    #         if token in key_dictionary:
    #             search_elements.append(key_dictionary[token])

    search_elements = np.random.choice(key_values, search_size, p=actual_frequencies)
    # np.random.shuffle(search_elements)

    return key_values, search_elements, predicted_frequencies, actual_frequencies

def get_bbc_dataset_adversary(path_directory, search_size=150000, dictionary_size=2000, train_size=400,
                              adversary_ratio=0, scale_factor=1, shuffle=False, __print__=False):
    # folder path
    # path_directory = "dataset\\bbc\\business\\"
    np.random.seed(42)
    files = path_directory + "*.txt"
    files = glob.glob(files)
    if shuffle:
        np.random.shuffle(files)
    n = len(files)
    n1 = train_size
    n2 = n
    print("{0}: Corpus Size: {1},  train size: {2}".format(path_directory, n, n1))
    data = {}
    search_data = {}
    for i in range(n):
        file_content = open(files[i]).read()
        tokens = word_tokenize(file_content)
        for token in tokens:
            if i < n1:
                if token not in data:
                    data[token] = 1
                else:
                    data[token] += 1
            if token not in search_data:
                search_data[token] = 1
            else:
                search_data[token] += 1

    common_items = {}
    for key in data:
        common_items[key] = search_data[key]

    all_tokens = np.sum(list(common_items.values()))

    # search_rankings = sorted(common_items.items(), key=lambda x: x[1], reverse=True)

    high_frequent_items = sorted(common_items.items(), key=lambda x: x[1], reverse=True)[:dictionary_size]
    print("Number of items: ", len(data.keys()))
    print("Number of high frequent items: ", len(high_frequent_items))
    key_values_string = []
    predicted_frequencies = []
    actual_frequencies = []
    for (key, freq) in high_frequent_items:
        key_values_string.append(key)
        actual_frequencies.append(search_data[key] ** scale_factor)

    actual_frequencies = np.array(actual_frequencies) / np.sum(actual_frequencies)

    sorted_key_elements = sorted(key_values_string)
    key_dictionary = {}

    for i in range(len(sorted_key_elements)):
        key_dictionary[sorted_key_elements[i]] = i

    key_values = []
    for key in key_values_string:
        key_values.append(key_dictionary[key])

    adversary_counts = {}
    for i in range(len(high_frequent_items)):
        key = high_frequent_items[i][0]
        # adversary_key = high_frequent_items[len(high_frequent_items) - i - 1][0]
        key_rank = key_dictionary[key]
        # adversary_count = common_items[adversary_key]
        adversary_count = high_frequent_items[key_rank][1]
        adversary_counts[key] = adversary_count

    for key in adversary_counts:
        ad_cnt = adversary_ratio * adversary_counts[key]
        normal_count = (1 - adversary_ratio) * common_items[key]
        common_items[key] = ad_cnt + normal_count

    for (key, freq) in high_frequent_items:
        predicted_frequencies.append(common_items[key] ** scale_factor)

    predicted_frequencies = np.array(predicted_frequencies) / np.sum(predicted_frequencies)

    # search_elements = []
    # for i in range(n):
    #     file_content = open(files[i]).read()
    #     tokens = word_tokenize(file_content)
    #     for token in tokens:
    #         if token in key_dictionary:
    #             search_elements.append(key_dictionary[token])

    if search_size > 0:
        search_elements = np.random.choice(key_values, search_size, p=actual_frequencies)
    else:
        search_elements = np.random.choice(key_values, all_tokens, p=actual_frequencies)

    # np.random.shuffle(search_elements)

    return key_values, search_elements, predicted_frequencies, actual_frequencies


def get_bbc_dataset_adversary_v2(path_directory, search_size=150000, dictionary_size=2000,
                              adversary_ratio=0, scale_factor=1, shuffle=False, __print__=False):
    # folder path
    # path_directory = "dataset\\bbc\\business\\"
    np.random.seed(42)
    files = path_directory + "*.txt"
    files = glob.glob(files)
    if shuffle:
        np.random.shuffle(files)
    n = len(files)
    n1 = n
    n2 = n
    print("{0}: Corpus Size: {1},  train size: {2}".format(path_directory, n, n1))
    data = {}
    search_data = {}
    all_tokens = 0
    for i in range(n):
        file_content = open(files[i]).read()
        tokens = word_tokenize(file_content)
        for token in tokens:
            all_tokens += 1
            if i < n1:
                if token not in data:
                    data[token] = 1
                else:
                    data[token] += 1
            if token not in search_data:
                search_data[token] = 1
            else:
                search_data[token] += 1

    common_items = {}
    for key in data:
        common_items[key] = search_data[key]

    # search_rankings = sorted(common_items.items(), key=lambda x: x[1], reverse=True)

    high_frequent_items = sorted(common_items.items(), key=lambda x: x[1], reverse=True)[:dictionary_size]
    print("Number of items: ", len(data.keys()))
    print("Number of high frequent items: ", len(high_frequent_items))
    key_values_string = []
    predicted_frequencies = []
    actual_frequencies = []
    for (key, freq) in high_frequent_items:
        key_values_string.append(key)
        actual_frequencies.append(search_data[key] ** scale_factor)

    actual_frequencies = np.array(actual_frequencies) / np.sum(actual_frequencies)

    sorted_key_elements = sorted(key_values_string)
    key_dictionary = {}

    for i in range(len(sorted_key_elements)):
        key_dictionary[sorted_key_elements[i]] = i

    key_values = []
    for key in key_values_string:
        key_values.append(key_dictionary[key])

    adversary_counts = {}
    for i in range(len(high_frequent_items)):
        key = high_frequent_items[i][0]
        # adversary_key = high_frequent_items[len(high_frequent_items) - i - 1][0]
        key_rank = key_dictionary[key]
        # adversary_count = common_items[adversary_key]
        adversary_count = high_frequent_items[key_rank][1]
        adversary_counts[key] = adversary_count

    for key in adversary_counts:
        ad_cnt = adversary_ratio * adversary_counts[key]
        normal_count = (1 - adversary_ratio) * common_items[key]
        common_items[key] = ad_cnt + normal_count

    for (key, freq) in high_frequent_items:
        predicted_frequencies.append(common_items[key] ** scale_factor)

    predicted_frequencies = np.array(predicted_frequencies) / np.sum(predicted_frequencies)

    # search_elements = []
    # for i in range(n):
    #     file_content = open(files[i]).read()
    #     tokens = word_tokenize(file_content)
    #     for token in tokens:
    #         if token in key_dictionary:
    #             search_elements.append(key_dictionary[token])
    if search_size >= 0:
        search_elements = np.random.choice(key_values, search_size, p=actual_frequencies)
    else:
        search_elements = np.random.choice(key_values, all_tokens, p=actual_frequencies)
    # np.random.shuffle(search_elements)

    return key_values, search_elements, predicted_frequencies, actual_frequencies, key_values_string

def get_bbc_dataset_adversary_Train(path_directory, key_values_string, search_size=150000, train_size=400,
                              adversary_ratio=0, scale_factor=1, shuffle=False, __print__=False):
    # folder path
    # path_directory = "dataset\\bbc\\business\\"
    np.random.seed(42)
    files = path_directory + "*.txt"
    files = glob.glob(files)
    if shuffle:
        np.random.shuffle(files)
    n = len(files)
    n1 = train_size
    n2 = n
    print("{0}: Corpus Size: {1},  train size: {2}".format(path_directory, n, n1))
    data = {}
    search_data = {}
    for i in range(n):
        file_content = open(files[i]).read()
        tokens = word_tokenize(file_content)
        for token in tokens:
            if i < n1:
                if token not in data:
                    data[token] = 1
                else:
                    data[token] += 1
            if token not in search_data:
                search_data[token] = 1
            else:
                search_data[token] += 1

    common_items = {}
    for key in key_values_string:
        if key in data:
            common_items[key] = search_data[key]
        else:
            data[key] = 3
            common_items[key] = 3

    # search_rankings = sorted(common_items.items(), key=lambda x: x[1], reverse=True)

    high_frequent_items = sorted(common_items.items(), key=lambda x: x[1], reverse=True)
    print("Number of items: ", len(data.keys()))
    print("Number of high frequent items: ", len(high_frequent_items))
    key_values_string = []
    predicted_frequencies = []

    for (key, freq) in high_frequent_items:
        key_values_string.append(key)


    sorted_key_elements = sorted(key_values_string)
    key_dictionary = {}

    for i in range(len(sorted_key_elements)):
        key_dictionary[sorted_key_elements[i]] = i

    adversary_counts = {}
    for i in range(len(high_frequent_items)):
        key = high_frequent_items[i][0]
        # adversary_key = high_frequent_items[len(high_frequent_items) - i - 1][0]
        key_rank = key_dictionary[key]
        # adversary_count = common_items[adversary_key]
        adversary_count = high_frequent_items[key_rank][1]
        adversary_counts[key] = adversary_count

    for key in adversary_counts:
        ad_cnt = adversary_ratio * adversary_counts[key]
        normal_count = (1 - adversary_ratio) * common_items[key]
        common_items[key] = ad_cnt + normal_count

    for (key, freq) in high_frequent_items:
        predicted_frequencies.append(common_items[key] ** scale_factor)

    predicted_frequencies = np.array(predicted_frequencies) / np.sum(predicted_frequencies)


    return predicted_frequencies