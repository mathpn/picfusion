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


class CompositeIndex:
    def __init__(self, img_ids: list[str], tags: list[set[str]], features: np.ndarray):
        self.img_ids = img_ids
        self.tags = tags
        self.features = features

    def find_knn_combined(
        self, query_tags: Iterable[str], query_feat: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        tag_overlap = self.get_tag_overlap(query_tags)
        tag_overlap = np.array(tag_overlap)
        clip_sim = self.get_clip_sim(query_feat)
        avg_sim = (tag_overlap + clip_sim) / 2
        sort_idx = np.argsort(-avg_sim)
        return [(self.img_ids[i], avg_sim[i]) for i in sort_idx[:top_k]]

    def find_knn_tags(self, query_tags: Iterable[str], top_k: int) -> list[tuple[str, float]]:
        tag_overlap = self.get_tag_overlap(query_tags)
        sort_idx = argsort(tag_overlap, ascending=False)
        return [(self.img_ids[i], tag_overlap[i]) for i in sort_idx[:top_k]]

    def find_knn_clip(self, query_feat: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        clip_sim = self.get_clip_sim(query_feat)
        sort_idx = np.argsort(-clip_sim)
        return [(self.img_ids[i], clip_sim[i]) for i in sort_idx[:top_k]]

    def get_tag_overlap(self, query_tags: Iterable[str]) -> list[float]:
        if not isinstance(query_tags, set):
            query_tags = set(query_tags)
        return [len(query_tags & tags) / len(tags) for tags in self.tags]

    def get_clip_sim(self, query_feat: np.ndarray) -> np.ndarray:
        return (query_feat.reshape(1, -1) @ self.features.T)[0, :]
