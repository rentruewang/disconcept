from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import loguru
from sentence_transformers import SentenceTransformer
from torch import Tensor, cuda
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .relations import RawRelation, Relation

_DEVICE = "cuda" if cuda.is_available() else "cpu"
_LOGGER = loguru.logger


class RelationExtractor:
    def __init__(
        self,
        *,
        relation_model: str = "Babelscape/rebel-large",
        # sentence_model: str = "all-MiniLM-L6-v2",
        sentence_model: str = "all-mpnet-base-v2",
    ) -> None:
        _LOGGER.warning(f"using device: {_DEVICE}")
        self.relation = AutoModelForSeq2SeqLM.from_pretrained(relation_model).to(
            _DEVICE
        )
        self.sentence = SentenceTransformer(sentence_model).to(_DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(relation_model)

    def __call__(self, text: str, /) -> list[Relation]:
        return self.extract(text)

    def encode(self, text: str) -> Mapping[str, Tensor]:
        _LOGGER.info(f"calling encode with text. {text}")
        return self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )

    def transform(self, encoded: Any) -> Tensor:
        kwargs = {
            "max_length": 512,
            "length_penalty": 0,
            "num_beams": 8,
            "num_return_sequences": 8,
        }
        encoded = encoded.to(_DEVICE)
        _LOGGER.info(f"calling transform {kwargs}")
        return self.relation.generate(
            **encoded,
            **kwargs,
        ).cpu()

    def decode(self, outputs: Tensor) -> Sequence[str]:
        _LOGGER.info("decoding")
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract(self, text: str) -> list[Relation]:
        encoded = self.encode(text)
        parsed = self.transform(encoded)
        decoded = self.tokenizer.batch_decode(parsed, skip_special_tokens=False)
        relations = [self.extract_lm_output(line) for line in decoded]
        combined = sum(relations, [])
        _LOGGER.info(f"encode into raw")
        out = [self.encode_relation(raw) for raw in combined]
        _LOGGER.info("extract done")
        return out

    def extract_lm_output(self, text: str) -> list[RawRelation]:
        relations = []
        relation = subject = relation = object_ = ""
        text = text.strip()
        current = "x"
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        for token in text_replaced.split():
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    relations.append(
                        RawRelation(subject.strip(), relation.strip(), object_.strip())
                    )
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    relations.append(
                        RawRelation(subject.strip(), relation.strip(), object_.strip())
                    )
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token
        if subject != "" and relation != "" and object_ != "":
            relations.append(
                RawRelation(subject.strip(), relation.strip(), object_.strip())
            )
        return relations

    def encode_relation(self, raw: RawRelation) -> Relation:
        return Relation.encode(raw, embedder=self.sentence)
