from __future__ import annotations

from typing import NamedTuple, Protocol, TypeVar, runtime_checkable

from numpy import ndarray
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer  # type: ignore

T = TypeVar("T")


@runtime_checkable
class Subject(Protocol[T]):
    sub: T


@runtime_checkable
class Related(Protocol[T]):
    rel: T
    obj: T


class RawRelation(NamedTuple):
    sub: str
    rel: str
    obj: str


class WithEmb(NamedTuple):
    text: str
    emb: NDArray

    def __eq__(self, other):
        if isinstance(other, str):
            return self.text == other

        raise NotImplementedError

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return repr(self.text)

    @property
    def dim(self) -> int:
        assert self.emb.ndim == 1
        return self.emb.size

    @classmethod
    def encode(cls, text: str, embedder: SentenceTransformer) -> WithEmb:
        assert isinstance(text, str)
        emb = embedder.encode(text)
        assert isinstance(emb, ndarray)
        return WithEmb(text, emb)


class Relation(NamedTuple):
    sub: WithEmb
    rel: WithEmb
    obj: WithEmb

    @classmethod
    def encode(cls, raw: RawRelation, embedder: SentenceTransformer) -> Relation:
        sub, rel, obj = raw
        return cls(
            WithEmb.encode(sub, embedder),
            WithEmb.encode(rel, embedder),
            WithEmb.encode(obj, embedder),
        )
