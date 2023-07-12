from typing import Iterable

def argsort(seq, ascending: bool = True):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    sorted_seq = sorted(range(len(seq)), key= seq.__getitem__)
    if ascending:
        return sorted_seq
    return list(reversed(sorted_seq))


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
