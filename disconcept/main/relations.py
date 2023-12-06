from __future__ import annotations

import wikipedia
from rich import print

from disconcept.core import RelationExtractor


def main():
    napoleon = wikipedia.page("Napoleon")
    text = napoleon.summary

    out = RelationExtractor()(text)

    print(out)
