from __future__ import annotations

import collections
import functools
from typing import Dict, NamedTuple

import alive_progress as prog
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


class Graph:
    def __init__(self, contexts: list[Context], device: str = None) -> None:
        self.contexts = contexts

        subject_set = set(c.sub.text for c in contexts)
        relation_set = set(s.text for s in sum([c.rel for c in contexts], []))
        object_set = set(s.text for s in sum([c.obj for c in contexts], []))

        self._subjects = sorted(subject_set)
        self._relations = sorted(relation_set)
        self._objects = sorted(object_set)

        self._subject_idx = {sub: i for i, sub in enumerate(self._subjects)}
        self._relation_idx = {rel: i for i, rel in enumerate(self._relations)}
        self._object_idx = {obj: i for i, obj in enumerate(self._objects)}

        self._subject_embed = {c.sub.text: c.sub.emb for c in contexts}
        self._relation_embed = {
            r.text: r.emb for r in sum([c.rel for c in contexts], [])
        }
        self._object_embed = {o.text: o.emb for o in sum([c.obj for c in contexts], [])}

    def subject_global_index(self, subject: str) -> int:
        return self._subject_idx[subject]

    def relation_global_index(self, relation: str) -> int:
        return self._relation_idx[relation] + len(self._subject_idx)

    def object_global_index(self, object: str) -> int:
        return (
            self._object_idx[object] + len(self._subject_idx) + len(self._relation_idx)
        )

    def embedding_from_index(self, index: int) -> NDArray:
        if index < len(self._subject_idx):
            emb = self._subject_embed[self._subjects[index]]
        elif index < len(self._subject_idx) + len(self._relation_idx):
            emb = self._relation_embed[self._relations[index - len(self._subject_idx)]]
        else:
            emb = self._object_embed[
                self._objects[index - len(self._subject_idx) - len(self._relation_idx)]
            ]
        assert emb.shape == (self.embed_dim,)

        if index < len(self._subject_idx):
            return np.concatenate([emb, np.zeros([2 * self.embed_dim])])
        elif index < len(self._subject_idx) + len(self._relation_idx):
            return np.concatenate(
                [np.zeros([self.embed_dim]), emb, np.zeros([self.embed_dim])]
            )
        else:
            return np.concatenate([np.zeros([2 * self.embed_dim]), emb])

    @property
    def embed_dim(self):
        return 768

    @functools.cached_property
    def x(self) -> NDArray:
        out = []
        for i in prog.alive_it(
            range(len(self._objects) + len(self._subjects) + len(self._relations))
        ):
            out.append(self.embedding_from_index(i))
        return np.array(out)

    @functools.cached_property
    def edge_index(self) -> NDArray:
        edges = []

        for ctx in prog.alive_it(self.contexts):
            sub = self.subject_global_index(ctx.sub.text)
            rels = [self.relation_global_index(r.text) for r in ctx.rel]
            objs = [self.object_global_index(o.text) for o in ctx.obj]

            for r, o in zip(rels, objs):
                edges.append((sub, r))
                edges.append((r, o))

        return np.array(edges)

    def group_index(self, assignment: Dict[str, int]):
        idx_group_mapping = {i: assignment[v] for v, i in self._subject_idx.items()}
        ordered_mapping = [idx_group_mapping[i] for i in range(len(self._subject_idx))]
        return np.concatenate(
            [ordered_mapping, [-100] * (len(self._relation_idx) + len(self._objects))]
        )


class GnnOutput(NamedTuple):
    output: Tensor
    direct: Tensor
    pooler: Tensor
    loss: Tensor
    acc_gcn: float
    acc_lin: float


GnnOutput = collections.namedtuple(
    "GnnOutput", ["output", "direct", "pooler", "loss", "acc"]
)


class Gnn(Module):
    def __init__(self, graph: Graph, assignment: Dict[str, int]) -> None:
        super().__init__()
        self.graph = graph
        self.norm = BatchNorm1d(num_features=self.graph.embed_dim * 3)
        self.gcn1 = GATConv(
            in_channels=self.graph.embed_dim * 3, out_channels=self.graph.embed_dim
        )
        self.gcn2 = GATConv(
            in_channels=self.graph.embed_dim * 4, out_channels=self.graph.embed_dim
        )
        self.pool_gcn = Linear(in_features=self.graph.embed_dim, out_features=features)

        self.direct = Linear(
            in_features=self.graph.embed_dim * 3, out_features=features
        )

        features = len(np.unique(list(assignment.values())))

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.assign = assignment
        self.activation = lambda x: x
        self.float().to(_DEVICE)

    def forward(self) -> GnnOutput:
        x = torch.from_numpy(self.graph.x).float().to(_DEVICE)
        edge_index = torch.from_numpy(self.graph.edge_index).T.long().to(_DEVICE)
        target = (
            torch.from_numpy(self.graph.group_index(self.assign)).long().to(_DEVICE)
        )

        assert edge_index.shape[0] == 2
        x = self.norm(x)

        direct = self.direct(x)
        out = self.gcn1(x, edge_index)
        out = self.activation(out)
        out = self.gcn2(torch.cat([out, x], dim=-1), edge_index)
        out = self.activation(out)

        pooler = self.pool_gcn(out)

        subjects = target >= 0

        assert pooler.shape == direct.shape

        ds = direct[subjects]
        ps = pooler[subjects]
        ts = target[subjects]

        loss = self.loss_fn(ps, ts) + self.loss_fn(ds, ts)

        with torch.no_grad():
            acc_gcn = (ps.argmax(-1) == ts).sum().item() / subjects.sum().item()
            acc_lin = (ds.argmax(-1) == ts).sum().item() / subjects.sum().item()

        return GnnOutput(
            output=out,
            direct=direct,
            pooler=pooler,
            loss=loss,
            acc_gcn=acc_gcn,
            acc_lin=acc_lin,
        )

    @classmethod
    def train(cls, graph: Graph, assignment: dict[str, int], *, cfg: DictConfig):
        epochs = int(cfg["ml"]["epochs"])
        lr = float(cfg["ml"]["lr"])
        moving_avg = int(cfg["ml"]["moving_avg"])

        gnn = cls(graph, assignment=assignment)
        classes = len(np.unique(list(assignment.values())))

        optimizer = Adam(gnn.parameters(), lr=lr)

        acc_gcn = None
        past_acc_gcn = collections.deque()
        for epoch in prog.alive_it(range(epochs)):
            if len(past_acc_gcn) >= moving_avg:
                past_acc_gcn.popleft()

            output: GnnOutput = gnn()

            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

            # Early stopping.
            if (
                epoch > (moving_avg + 1)
                and abs(sum(past_acc_gcn) / len(past_acc_gcn) - acc_gcn) < 1e-6
            ):
                break

            acc_gcn = output.acc_gcn
            acc_lin = output.acc_lin
            past_acc_gcn.append(acc_gcn)
        return {"acc_gcn": acc_gcn, "acc_lin": acc_lin, "classes": classes}
