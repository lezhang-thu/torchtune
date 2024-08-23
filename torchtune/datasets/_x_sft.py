# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class BioDataset(Dataset):

    def __init__(
        self,
        *,
        source: str,
        model_transform: Transform,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._model_transform = model_transform
        self._data = load_dataset(source, **load_dataset_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        tokenized_dict = self._model_transform(sample)

        # Wherever masked == False, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        masked = tokenized_dict.pop("mask")
        tokenized_dict["labels"] = list(
            np.where(
                masked,
                # debug: "mask" meaning
                tokenized_dict["gt"],
                CROSS_ENTROPY_IGNORE_IDX,
            ))
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])

        return tokenized_dict


def x_bio_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = None,
    column_map: Optional[Dict[str, str]] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> BioDataset:
    return BioDataset(
        source=source,
        model_transform=tokenizer,
        split=split,
        **load_dataset_kwargs,
    )


bio_dataset = partial(x_bio_dataset,
                      source="./train-split")
test_bio_dataset = partial(
    x_bio_dataset,
    source="./test-split",
    split="test",
)
