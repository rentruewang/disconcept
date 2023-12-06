from __future__ import annotations

from rich import print

from disconcept.core import find, union


def main():
    parent = {i: i for i in range(10)}

    find(parent, 1)
    union(parent, 1, 2)
    union(parent, 3, 4)
    union(parent, 5, 6)
    union(parent, 1, 6)

    print(parent)
    print(find(parent, 2))
    print(find(parent, 6))
    print(parent)
