from __future__ import annotations

import collections

import loguru

from .relations import Relation, WithEmb


class Context:
    """
    Context contains a word and all its relations.
    """

    def __init__(
        self,
        sub: WithEmb | None = None,
        rel_obj_set: list[tuple[WithEmb, WithEmb]] | None = None,
    ) -> None:
        self._sub = sub

        if rel_obj_set:
            self._rel_obj_set = rel_obj_set
        else:
            self._rel_obj_set = []

    def __repr__(self) -> str:
        return repr(
            {
                "sub": self.sub,
                "rel:obj": {r.text: o.text for r, o in zip(self.rel, self.obj)},
            }
        )

    @property
    def sub(self) -> WithEmb:
        assert self._sub is not None
        return self._sub

    @property
    def rel(self) -> list[WithEmb]:
        return [r for r, _ in self._rel_obj_set]

    @property
    def obj(self) -> list[WithEmb]:
        return [o for _, o in self._rel_obj_set]

    @classmethod
    def from_relations(cls, relations: list[Relation]) -> Context:
        assert len(set(rel.sub.text for rel in relations)) == 1, [
            rel.sub.text for rel in relations
        ]

        rel_obj: list[tuple[WithEmb, WithEmb]] = [
            (rel.rel, rel.obj) for rel in relations
        ]
        unique_rel_obj: list[tuple[WithEmb, WithEmb]] = []
        existed: set[tuple[str, str]] = set()

        for rel, obj in rel_obj:
            if (rel.text, obj.text) in existed:
                continue

            existed.add((rel.text, obj.text))
            unique_rel_obj.append((rel, obj))

        return cls(sub=relations[0].sub, rel_obj_set=unique_rel_obj)


def relations_to_contexts(relations: list[Relation]) -> list[Context]:
    loguru.logger.info(f"Converting relations of size {len(relations)}")
    by_name = collections.defaultdict(list)

    for rel in relations:
        by_name[rel.sub.text].append(rel)

    return [Context.from_relations(rel) for rel in by_name.values()]
