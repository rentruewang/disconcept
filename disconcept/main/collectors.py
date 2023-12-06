from __future__ import annotations

from rich import print, traceback

from disconcept.core import GoogleNewsCollector, WikipediaAugmenter


def main():
    traceback.install()

    gnc = GoogleNewsCollector()

    # out = gnc.recursive("jojo", 100)
    # print(len(out), out)

    wiki = WikipediaAugmenter()
    print(wiki.augment(["napoleon"], 100))
