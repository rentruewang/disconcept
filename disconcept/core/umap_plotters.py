from __future__ import annotations

from matplotlib import pyplot as plt
from numpy.typing import NDArray
from umap import UMAP, plot


class UmapPlotter:
    def __init__(self, contexts: NDArray, labels: NDArray) -> None:
        assert len(contexts) == len(labels)

        self.__fitted = False

        self.embeddings = contexts
        self.labels = labels

        self.umap = UMAP()

    def fit_transform(self):
        self.__fitted = True
        return self.umap.fit_transform(self.embeddings)

    def draw(self, filename: str):
        assert self.__fitted
        plot.points(self.umap, labels=self.labels)
        plt.savefig(filename)
