import pickle
from pathlib import Path

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


def run(threshold: float, cfg: DictConfig):
    retrieval_dump = Path(cfg["dumps"]["retrieval"])
    similarity_dump = Path(cfg["dumps"]["similarity"])

    limit = int(cfg["download"]["limit"])
    keyword = cfg["download"]["keyword"]

    if retrieval_dump.exists():
        _LOGGER.warning("loading retrieval from cache")
        with open(retrieval_dump, "rb") as f:
            retrieved_data = pickle.load(f)
    else:
        _LOGGER.warning("Downloading data from the internet")
        main = Downloader()
        retrieved_data = main.download(keyword, limit=limit)

        with open(retrieval_dump, "wb+") as f:
            pickle.dump(retrieved_data, f)

        print(len(retrieved_data))

    mst = ThresholdMST(retrieved_data)
    if similarity_dump.exists():
        _LOGGER.warning("loading similarity from cache")
        with open(similarity_dump, "rb") as f:
            similarity = pickle.load(f)
        mst.similarity = similarity
    else:
        # Evaluate it first, then it would be cached.
        _LOGGER.info("running mst...")
        _ = mst.similarity

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
