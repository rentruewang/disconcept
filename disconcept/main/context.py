from __future__ import annotations

import wikipedia
from rich import print

from disconcept.core import RelationExtractor, relations_to_contexts


def main():
    napoleon = wikipedia.page("Napoleon")
    text = napoleon.summary

    out = RelationExtractor()(text)

    filtered = [o for o in out if o.sub.text == "Napoleon Bonaparte"]
    print(filtered)

    ctx = relations_to_contexts(filtered)

    print(ctx)
