from typing import Iterable

import numpy as np


def argsort(seq, ascending: bool = True):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    sorted_seq = sorted(range(len(seq)), key=seq.__getitem__)
    if ascending:
        return sorted_seq
    return list(reversed(sorted_seq))


def cos_sim(a, b):
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


class TagIndex:
    def __init__(self, img_ids: list[str], tags: list[set[str]]):
        self.img_ids = img_ids
        self.tags = tags

    def find_knn(self, query_tags: Iterable[str], top_k: int) -> list[str]:
        if not isinstance(query_tags, set):
            query_tags = set(query_tags)
        overlap = [len(query_tags & tags) for tags in self.tags]
        sort_idx = argsort(overlap, ascending=False)
        return [self.img_ids[i] for i in sort_idx[:top_k]]


class ClipIndex:
    def __init__(self, img_ids: list[str], features: np.ndarray):
        self.img_ids = img_ids
        self.features = features

    def find_knn(self, query_feats: np.ndarray, top_k: int) -> list[str]:
        # XXX features must be of unit norm!
        sim = query_feats.reshape(1, -1) @ self.features.T
        sort_idx = np.argsort(-sim[0])
        return [self.img_ids[i] for i in sort_idx[:top_k]]
