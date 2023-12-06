from __future__ import annotations

import pickle

import numpy as np

from disconcept.core import Gnn, Graph, UmapPlotter


def main():
    contexts = pickle.load(open("retrieval_dump.pkl", "rb"))
    graph = Graph(contexts)
    gnn = Gnn(graph)
    out = gnn()
    print(out.shape)

    up = UmapPlotter(
        out.detach().cpu().numpy(), labels=np.random.randint(0, 3, [len(out)])
    )
    up.fit_transform()
    up.draw()
