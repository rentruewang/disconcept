from __future__ import annotations

import itertools

import numpy as np
from alive_progress import alive_it
from loguru import logger
from numpy.typing import NDArray
from scipy import optimize

from .contexts import Context


def find(parent: dict, /, key):
    if (par := parent[key]) == key:
        return key

    found = find(parent, key=par)
    parent[key] = found
    return found


def union(parent: dict, /, a, b) -> None:
    if parent[a] == parent[b]:
        return

    par_a = find(parent, key=a)
    par_b = find(parent, key=b)

    parent[par_b] = par_a


def ctx_cos_sim(
    rel_i: NDArray, rel_j: NDArray, obj_i: NDArray, obj_j: NDArray
) -> float:
    assert rel_i.shape[1] == rel_j.shape[1]
    rel_i /= (rel_i**2).sum(-1, keepdims=True)
    rel_j /= (rel_j**2).sum(-1, keepdims=True)

    cos_rel = rel_i @ rel_j.T
    assert (
        np.max(cos_rel) <= 1 + 1e-5 and np.min(cos_rel) >= -1 - 1e-5
    ), f"cos in [-1, 1], got {(np.max(cos_rel), np.min(cos_rel))}"

    assert obj_i.shape[1] == obj_j.shape[1]
    obj_i /= (obj_i**2).sum(-1, keepdims=True)
    obj_j /= (obj_j**2).sum(-1, keepdims=True)

    src, tgt = optimize.linear_sum_assignment(cos_rel, maximize=True)

    mapped_i = obj_i[src]
    mapped_j = obj_j[tgt]

    assert mapped_i.shape == mapped_j.shape

    out = (mapped_i * mapped_j).sum()
    assert (
        -min(len(mapped_i), len(mapped_j)) - 1e-5
        <= out
        <= min(len(mapped_i), len(mapped_j)) + 1e-5
    ), "result in [-1, 1]"
    return out


class ThresholdMST:
    def __init__(self, contexts: list[Context]) -> None:
        self.contexts = contexts
        self.__similarity = None

    def _get_sim(self) -> NDArray:
        if self.__similarity is not None:
            return self.__similarity

        l = len(self.contexts)
        sim = np.zeros([l, l])

        relations = [np.array([r.emb for r in ctx.rel]) for ctx in self.contexts]
        objects = [np.array([r.emb for r in ctx.obj]) for ctx in self.contexts]

        for i, j in alive_it(itertools.product(range(l), range(l)), total=l * l):
            if i > j:
                continue

            if i == j:
                sim[i, j] = 0

            sim[i, j] = sim[j, i] = ctx_cos_sim(
                relations[i], relations[j], objects[i], objects[j]
            )

        logger.info("max: {} min: {}".format(np.max(sim), np.min(sim)))
        return sim

    def _set_sim(self, value: NDArray):
        self.__similarity = value

    similarity = property(_get_sim, _set_sim)

    def group(self, threshold: float):
        l = len(self.contexts)
        cost = self.similarity

        logger.info("min cost: {} max cost: {}".format(np.min(cost), np.max(cost)))

        logger.info("sorting costs")
        sources, targets = np.unravel_index(np.argsort(cost, axis=None), cost.shape)
        sources = sources[::-1]
        targets = targets[::-1]
        logger.info(f"done")

        parents = {i: i for i in range(l)}

        logger.info("unioning")
        for i, j in zip(sources, targets):
            c = cost[i, j]
            if c < threshold:
                break

            union(parents, i, j)
        logger.info("done unioning")

        assignment = {self.contexts[i].sub.text: find(parents, i) for i in range(l)}
        logger.warning("classes = {}".format(len(set(assignment.values()))))

        values = np.array(list(assignment.values()))
        unique_values = np.unique(values)
        to_unique = {uni: idx for idx, uni in enumerate(unique_values)}

        return {key: to_unique[value] for key, value in assignment.items()}
