from __future__ import annotations

import wikipedia
from rich import print


def main():
    page = wikipedia.page("Napoleon")
    print(page.summary)
    print(page)
