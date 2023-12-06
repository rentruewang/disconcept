import pickle
from pathlib import Path
from typing import Callable

import hydra
import loguru
import numpy as np
from omegaconf import DictConfig

from disconcept import Downloader, Gnn, Graph, ThresholdMST

_LOGGER = loguru.logger


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    results = Path(cfg["dumps"]["results"])

    result = {}
    for threshold in reversed(np.linspace(0, 10, 100)):
        _LOGGER.warning(f"threshold = {threshold}")
        result[threshold] = run(threshold, cfg=cfg)
        _LOGGER.warning(f"result = {result[threshold]}")

    with open(results, "wb+") as f:
        pickle.dump(result, f)


def _load_cache_or_run(path: Path, func: Callable[[], None]):
    if path.exists():
        _LOGGER.warning(f"loading from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    result = func()

    with open(path, "wb+") as f:
        _LOGGER.warning(f"dumping to {path}")
        pickle.dump(result, f)

    return result


def run(threshold: float, cfg: DictConfig):
    retrieval_dump = Path(cfg["dumps"]["retrieval"])
    similarity_dump = Path(cfg["dumps"]["similarity"])

    limit = int(cfg["download"]["limit"])
    keyword = cfg["download"]["keyword"]

    _LOGGER.warning("Performing downloading...")

    # Lazily download data if cache is not found.
    download = lambda: Downloader().download(keyword=keyword, limit=limit)
    retrieved_data = _load_cache_or_run(retrieval_dump, download)

    mst = ThresholdMST(retrieved_data)

    _LOGGER.warning("Computing similarity...")

    # Lazily compute MST if cache is not found.
    comp_sim = lambda: mst.similarity
    mst.similarity = _load_cache_or_run(similarity_dump, comp_sim)

    _LOGGER.info("grouping...")
    assignment = mst.group(threshold)

    _LOGGER.info("training...")
    graph = Graph(contexts=retrieved_data)

    _LOGGER.info(f"number of subjects: {len(graph._subjects)}")
    _LOGGER.info(f"number of relations: {len(graph._relations)}")
    _LOGGER.info(f"number of objects: {len(graph._objects)}")

    _LOGGER.info(f"number of contexts: {len(graph.contexts)}")
    return Gnn.train(graph, assignment, cfg=cfg)


if __name__ == "__main__":
    main()
