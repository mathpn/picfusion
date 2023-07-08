import random
from timeit import timeit

import numpy as np
from pympler import asizeof


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def main():
    with open("./ram/data/ram_tag_list.txt", "r") as f:
        words = f.read().splitlines()

    word_map = {word: i for i, word in enumerate(words)}
    n_sets = 100

    random.shuffle(words)
    for n_words in (1, 10, 100, 1000):
        base_set = set(words[:n_words])
        base_int_set = {word_map[word] for word in words[:n_words]}
        base_numpy = np.zeros(len(word_map), dtype=bool)
        for word in words[:n_words]:
            base_numpy[word_map[word]] = True

        sets = [set(random.sample(words, random.randint(5, 50))) for _ in range(n_sets)]
        int_sets = [{word_map[word] for word in word_set} for word_set in sets]
        numpy_sets = [np.zeros_like(base_numpy, dtype=bool) for _ in range(n_sets)]
        for i, word_set in enumerate(sets):
            for word in word_set:
                numpy_sets[i][word_map[word]] = True

        stacked_numpy_sets = np.stack(numpy_sets, axis=0)

        def numpy_overlap():
            overlaps = [(base_numpy & numpy_set).sum() for numpy_set in numpy_sets]
            return np.argsort(overlaps)

        def numpy_stack_overlap():
            overlaps = (base_numpy.reshape(1, -1) & stacked_numpy_sets).sum(axis=1)
            return np.argsort(overlaps)

        def set_overlap():
            overlaps = [base_set & word_set for word_set in sets]
            return np.argsort(overlaps)

        def set_overlap_2():
            overlaps = [base_set & word_set for word_set in sets]
            return argsort(overlaps)

        def set_int_overlap():
            overlaps = [base_int_set & int_set for int_set in int_sets]
            return np.argsort(overlaps)

        def set_int_overlap_2():
            overlaps = [base_int_set & int_set for int_set in int_sets]
            return argsort(overlaps)

        n_runs = 1_000_000 // n_sets
        conversion_factor = 1_000_000 / n_runs
        print(f"\nnumber of words in base set: {n_words}")
        print(f"numpy overlap:       {timeit(numpy_overlap, number=n_runs) * conversion_factor:.2f} us")
        print(f"numpy stack overlap: {timeit(numpy_stack_overlap, number=n_runs) * conversion_factor:.2f} us")
        print(f"set overlap:         {timeit(set_overlap, number=n_runs) * conversion_factor:.2f} us")
        print(f"set overlap 2:       {timeit(set_overlap_2, number=n_runs) * conversion_factor:.2f} us")
        print(f"set int overlap:     {timeit(set_int_overlap, number=n_runs) * conversion_factor:.2f} us")
        print(f"set int overlap 2:   {timeit(set_int_overlap_2, number=n_runs) * conversion_factor:.2f} us")
        print()
        print(f"numpy size: {asizeof.asizeof(base_numpy)} bytes")
        print(f"set size: {asizeof.asizeof(base_set)} bytes")
        print(f"set int size: {asizeof.asizeof(base_int_set)} bytes")

if __name__ == "__main__":
    main()
