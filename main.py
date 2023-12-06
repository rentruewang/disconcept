import collections
import pickle
from pathlib import Path

import alive_progress as prog
import hydra
import loguru
import numpy as np
from omegaconf import DictConfig
from torch.optim import Adam

from disconcept import Downloader, Gnn, Graph, ThresholdMST

_LOGGER = loguru.logger


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    results = Path(cfg["paths"]["results"])

    result = {}
    for threshold in reversed(np.linspace(0, 10, 100)):
        _LOGGER.warning(f"threshold = {threshold}")
        result[threshold] = run(threshold, cfg=cfg)
        _LOGGER.warning(f"result = {result[threshold]}")

    with open(results, "wb+") as f:
        pickle.dump(result, f)


def run(threshold: float, cfg: DictConfig):
    epochs = int(cfg["ml"]["epochs"])
    lr = float(cfg["ml"]["lr"])

    retrieval_dump = Path(cfg["paths"]["retrieval"])
    similarity_dump = Path(cfg["paths"]["similarity"])

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
    gnn = Gnn(graph, assignment=assignment)
    # print(gnn)
    classes = len(np.unique(list(assignment.values())))

    optimizer = Adam(gnn.parameters(), lr=lr)

    acc = None
    past_acc = collections.deque()
    for epoch in prog.alive_it(range(epochs)):
        if len(past_acc) >= 100:
            past_acc.popleft()

        output = gnn()

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

        if epoch > 101 and abs(sum(past_acc) / len(past_acc) - acc) < 1e-6:
            break

        acc = output.acc
        past_acc.append(acc)
        # print(output.loss.item(), output.acc)

    return {"acc": acc, "classes": classes}


if __name__ == "__main__":
    main()
