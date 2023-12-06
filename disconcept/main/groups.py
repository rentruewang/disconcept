from __future__ import annotations

import wikipedia
from rich import print

from disconcept.core import RelationExtractor, relations_to_contexts


def main():
    napoleon = wikipedia.page("Napoleon")
    napoleon_text = napoleon.summary

    out = RelationExtractor()(napoleon_text)

    filtered = [o for o in out if o.sub.text == "Napoleon Bonaparte"]
    print(filtered)

    napoleon_ctx = relations_to_contexts(filtered)

    print(napoleon_ctx)

    alexander = wikipedia.page("Alexander the Great")
    alexander_text = alexander.summary

    out = RelationExtractor()(alexander_text)

    filtered = [o for o in out if o.sub.text == "Alexander the Great"]
    print(filtered)

    alexander_ctx = relations_to_contexts(filtered)

    print(alexander_ctx)
