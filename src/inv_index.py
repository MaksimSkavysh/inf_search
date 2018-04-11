from collections import defaultdict


def get_inv_index(documents):
    inv_index = defaultdict(list)
    N = 0
    total_words = 0
    for index in documents:
        N = N + 1
        document = documents[index]
        total_words = total_words + len(document)
        for token in document:
            if not inv_index[token]:
                inv_index[token] = []
            if index not in inv_index[token]:
                inv_index[token].append(index)
    L = total_words / N
    return inv_index, N, L
