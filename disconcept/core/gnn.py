from __future__ import annotations

import collections
import functools
from typing import NamedTuple

import alive_progress as prog
import loguru
import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor, cuda
from torch.nn import BatchNorm1d, CrossEntropyLoss, Linear, Module
from torch.optim import Adam
from torch_geometric.nn import GATConv

from .contexts import Context

_DEVICE = "cuda" if cuda.is_available() else "cpu"
_LOGGER = loguru.logger
_EMBED_DIM = 768
_IGNORE = -100


class Graph:
    def __init__(self, contexts: list[Context]) -> None:
        self.contexts = contexts

        ctx_rels = sum([c.rel for c in contexts], [])
        ctx_objs = sum([c.obj for c in contexts], [])

        subject_set = set(c.sub.text for c in contexts)
        relation_set = set(s.text for s in ctx_rels)
        object_set = set(s.text for s in ctx_objs)

        self._subjects = sorted(subject_set)
        self._relations = sorted(relation_set)
        self._objects = sorted(object_set)

        self._sub_idx = {sub: i for i, sub in enumerate(self._subjects)}
        self._rel_idx = {rel: i for i, rel in enumerate(self._relations)}
        self._obj_idx = {obj: i for i, obj in enumerate(self._objects)}

        self._sub_emb = {c.sub.text: c.sub.emb for c in contexts}
        self._rel_emb = {r.text: r.emb for r in ctx_rels}
        self._obj_emb = {o.text: o.emb for o in ctx_objs}

    def subject_global_index(self, subject: str) -> int:
        return self._sub_idx[subject]

    def relation_global_index(self, relation: str) -> int:
        return self._rel_idx[relation] + self.num_subs

    def object_global_index(self, object: str) -> int:
        return self._obj_idx[object] + self.num_subs + self.num_rels

    def embedding_from_index(self, index: int) -> NDArray:
        if index < self.num_subs:
            emb = self._sub_emb[self._subjects[index]]
        elif index < self.num_subs + self.num_rels:
            emb = self._rel_emb[self._relations[index - self.num_subs]]
        else:
            emb = self._obj_emb[self._objects[index - self.num_subs - self.num_rels]]
        assert emb.shape == (_EMBED_DIM,)

        paddings = np.zeros([_EMBED_DIM])
        if index < self.num_subs:
            return np.concatenate([emb, paddings, paddings])
        elif index < self.num_subs + self.num_rels:
            return np.concatenate([paddings, emb, paddings])
        else:
            return np.concatenate([paddings, paddings, emb])

    @property
    def num_subs(self):
        return len(self._subjects)

    @property
    def num_rels(self):
        return len(self._relations)

    @property
    def num_objs(self):
        return len(self._objects)

    @functools.cached_property
    def x(self) -> NDArray:
        _LOGGER.info("Constructing x")
        out = []
        for i in range(len(self._objects) + len(self._subjects) + len(self._relations)):
            out.append(self.embedding_from_index(i))
        return np.array(out)

    @functools.cached_property
    def edge_index(self) -> NDArray:
        _LOGGER.info("Constructing edges")
        edges = []
        for ctx in self.contexts:
            sub = self.subject_global_index(ctx.sub.text)
            rels = [self.relation_global_index(r.text) for r in ctx.rel]
            objs = [self.object_global_index(o.text) for o in ctx.obj]

            for r, o in zip(rels, objs):
                edges.append((sub, r))
                edges.append((r, o))

        return np.array(edges)

    def group_index(self, assignment: dict[str, int]):
        idx_group_mapping = {i: assignment[v] for v, i in self._sub_idx.items()}
        ordered_mapping = [idx_group_mapping[i] for i in range(self.num_subs)]
        return np.concatenate(
            [ordered_mapping, [_IGNORE] * (self.num_rels + self.num_objs)]
        )


class GnnOutput(NamedTuple):
    output: Tensor
    pooler: Tensor
    loss: Tensor
    acc: float


class Gnn(Module):
    def __init__(self, graph: Graph, assignment: dict[str, int]) -> None:
        super().__init__()

        features = len(set(assignment.values()))

        self.graph = graph
        self.norm = BatchNorm1d(num_features=_EMBED_DIM * 3)
        self.gcn1 = GATConv(in_channels=_EMBED_DIM * 3, out_channels=_EMBED_DIM)
        self.gcn2 = GATConv(in_channels=_EMBED_DIM * 4, out_channels=_EMBED_DIM)
        self.pooler = Linear(in_features=_EMBED_DIM, out_features=features)

        self.direct = Linear(in_features=_EMBED_DIM * 3, out_features=features)

        self.loss_fn = CrossEntropyLoss(ignore_index=_IGNORE)
        self.assign = assignment
        self.float().to(_DEVICE)

    def forward(self) -> GnnOutput:
        # Initializing variables.
        x = torch.tensor(self.graph.x).float().to(_DEVICE)
        edge_index = torch.tensor(self.graph.edge_index.T).long().to(_DEVICE)
        target = torch.tensor(self.graph.group_index(self.assign)).long().to(_DEVICE)

        # Forwarding.
        assert edge_index.shape[0] == 2
        x = self.norm(x)

        direct = self.direct(x)
        out = self.gcn1(x, edge_index)
        out = self.gcn2(torch.cat([out, x], dim=-1), edge_index)
        pooler = self.pooler(out)
        assert pooler.shape == direct.shape

        # Filtering.
        subjects = target >= 0
        ds = direct[subjects]
        ps = pooler[subjects]
        ts = target[subjects]
        assert subjects.sum().item() == self.graph.num_subs

        loss = self.loss_fn(ps, ts) + self.loss_fn(ds, ts)

        with torch.no_grad():
            acc = (ps.argmax(-1) == ts).sum().item() / self.graph.num_subs

        return GnnOutput(output=out, pooler=pooler, loss=loss, acc=acc)

    @classmethod
    def experiment(cls, graph: Graph, assignment: dict[str, int], *, cfg: DictConfig):
        epochs = int(cfg["ml"]["epochs"])
        lr = float(cfg["ml"]["lr"])
        moving_avg = int(cfg["ml"]["moving_avg"])

        gnn = cls(graph, assignment=assignment)
        classes = len(np.unique(list(assignment.values())))

        optimizer = Adam(gnn.parameters(), lr=lr)

        acc = None
        moving_average_queue = collections.deque()
        for epoch in prog.alive_it(range(epochs)):
            if len(moving_average_queue) >= moving_avg:
                moving_average_queue.popleft()

            output: GnnOutput = gnn()

            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

            # Early stopping.
            assert acc is not None
            if (
                epoch > (moving_avg + 1)
                and abs(sum(moving_average_queue) / len(moving_average_queue) - acc)
                < 1e-6
            ):
                break

            # Storing the outputs for the final returns.
            acc = output.acc
            moving_average_queue.append(acc)
        return {"acc_gcn": acc, "classes": classes}
