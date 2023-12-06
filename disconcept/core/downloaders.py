from __future__ import annotations

import alive_progress as prog
import loguru

from .collectors import GoogleNewsCollector, WikipediaAugmenter
from .contexts import Context, relations_to_contexts
from .extractors import RelationExtractor


class Downloader:
    def __init__(self) -> None:
        self.google_news = GoogleNewsCollector()
        self.wiki = WikipediaAugmenter()
        self.extractor = RelationExtractor()

    def download(self, keyword: str, /, limit: int) -> list[Context]:
        kws = self.google_news.document_keywords(keyword)
        augmented = self.wiki.augment(kws, limit=limit)

        output = []
        for text in prog.alive_it(augmented):
            output.extend(self.extractor(text))
            loguru.logger.info(f"Collected relations so far: {len(output)}")

        contexts = relations_to_contexts(output)
        return contexts
