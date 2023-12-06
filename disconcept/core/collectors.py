from __future__ import annotations

import collections

import alive_progress as prog
import loguru
import nltk
import wikipedia
from GoogleNews import GoogleNews
from joblib import Parallel, delayed
from newspaper import Article, ArticleException
from wikipedia import DisambiguationError, PageError


class GoogleNewsCollector:
    def __init__(self) -> None:
        # NLTK modules download.
        nltk.download("punkt")
        nltk.download("wordnet")

    @property
    def google_news(self):
        gn = GoogleNews()
        gn.set_lang("en")
        gn.set_period("7d")
        gn.set_encode("utf-8")
        return gn

    def search(self, query: str) -> list:
        gn = self.google_news

        gn.search(query)
        return gn.results(sort=True)

    def search_links(self, query: str):
        results = self.search(query)
        links = [r["link"] for r in results]
        return links

    def document_keywords(self, query: str):
        links = self.search_links(query)

        words = []
        bar = prog.alive_it(links)
        for link in bar:
            bar.text = f"document processing {link}"
            try:
                out = self.keywords(link)
                words.extend(out)
            except ArticleException:
                pass

        assert len(words)
        return words

    def keywords(self, link: str):
        art = Article(link)
        art.download()
        art.parse()
        art.nlp()
        return art.keywords


class WikipediaAugmenter:
    def __init__(self, processes: int = 16) -> None:
        self._augmented = set()
        self._processes = processes

    def augment(self, texts: list[str], limit: int) -> list[str]:
        text_queue = collections.deque(texts)

        while len(texts) < limit:
            text = text_queue.popleft()
            words = set(w.lower() for w in nltk.word_tokenize(text))

            not_augmented = [w for w in words if w not in self._augmented]
            self._augmented |= set(not_augmented)

            out = self.augment_in_parallel(not_augmented)
            not_null = [o for o in out if o]
            text_queue.extend(not_null)
            texts.extend(not_null)

        return texts

    def augment_in_parallel(self, texts: list[str]) -> list[str]:
        def gen():
            bar = prog.alive_it(texts)
            for item in bar:
                bar.text = f"augmenting texts: {item}"
                yield item

        return Parallel(n_jobs=self._processes, backend="threading")(
            delayed(self.augment_one)(t) for t in gen()
        )

    def augment_one(self, text: str) -> str:
        try:
            return wikipedia.page(text).summary
        except (PageError, DisambiguationError):
            return ""
        except KeyError:
            loguru.logger.warning("key error encountered")
            return ""
