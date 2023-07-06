import random


MERSENNES1 = [2**x - 1 for x in [17, 31, 127]]
MERSENNES2 = [2**x - 1 for x in [19, 67, 257]]


def simple_hash(int_list, prime1, prime2, prime3):
    """Compute a hash value from a list of integers and 3 primes"""
    result = 0
    for integer in int_list:
        result += ((result + integer + prime1) * prime2) % prime3
    return result


def hash1(int_list):
    return simple_hash(int_list, MERSENNES1[0], MERSENNES1[1], MERSENNES1[2])


def hash2(int_list):
    return simple_hash(int_list, MERSENNES2[0], MERSENNES2[1], MERSENNES2[2])


class BloomFilter:
    def __init__(self, size: int, n_probes: int):
        self.size = size
        self.n_probes = n_probes
        self.filter = 0

    def _get_probe_bits(self, key: str):
        chars = [ord(x) for x in key]
        hash_value1 = hash1(chars)
        hash_value2 = hash2(chars)
        probe_value = hash_value1
        for _ in range(self.n_probes):
            probe_value *= hash_value1
            probe_value += hash_value2
            probe_value %= MERSENNES1[2]
            yield probe_value % self.size

    def add(self, key: str):
        for bit in self._get_probe_bits(key):
            self.filter = self.filter | (1 << bit % self.size)

    def __contains__(self, key: str):
        return all(self.filter & (1 << bit) for bit in self._get_probe_bits(key))

    def __repr__(self):
        return bin(self.filter)


def main():
    with open("./ram/data/ram_tag_list.txt", "r") as f:
        words = f.read().splitlines()

    chunk_size = 20
    query_size = 5
    fps = []
    for _ in range(100):
        fp = 0
        bloom = BloomFilter(64, 3)
        random.shuffle(words)
        for word in words[:chunk_size]:
            bloom.add(word)

        n_trials = 0

        # for word in words[chunk_size:]:
        #     n_trials += 1
        #     if word in bloom:
        #         fp += 1

        n_trials = 0
        for j in range(query_size, len(words), query_size):
            n_trials += 1
            w = words[j : j + query_size]
            if all(ww in bloom for ww in w):
                fp += 1

        assert all(ww in bloom for ww in words[:chunk_size])
        fps.append(fp / n_trials)

    print(f"average false positive rate: {sum(fps) / len(fps):.8%}")


if __name__ == "__main__":
    main()
