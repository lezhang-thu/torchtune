# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.modules.tokenizers import ModelTokenizer

GENE2NUM = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
}

BASE = 4

MASK_PROB, RANDOM_TOKEN_PROB, LEAVE_UNMASKED_PROB = 0.15, 0.1, 0.1
#MASK_PROB, RANDOM_TOKEN_PROB, LEAVE_UNMASKED_PROB = 0.001, 0.1, 0.1

HYPHEN, BOS, CLS, LABEL = '_', '[BOS]', '[CLS]', 'X'

UN_DETECTED = set(['?_?', 'D_I', 'I_I', 'D_D'])

SPECIAL_TOKENS = {
    "?_?": 16,
    "D_I": 17,
    "I_I": 18,
    "D_D": 19,
    "[BOS]": 20,
    "[CLS]": 21,
}

LABEL2NUM = {
    'C': 0,
    'D': 1,
}


class x_Qwen2Tokenizer(ModelTokenizer):

    def __init__(
        self,
        max_seq_len: Optional[int] = None,
    ):
        self.max_seq_len = max_seq_len
        # except [BOS] & [CLS]
        self.max_range = np.arange(max_seq_len - 2)

    def tokenize_messages(
            self, sample: Dict[str, str]) -> Tuple[List[int], List[bool]]:
        ids = []
        mask_pos_okay = []
        for x in sample.values():
            if x in UN_DETECTED:
                ids.append(SPECIAL_TOKENS[x])
                mask_pos_okay.append(False)
            else:
                a, b = [GENE2NUM[_] for _ in x.split(HYPHEN)]
                ids.append(a * BASE + b)
                mask_pos_okay.append(True)

        gt = copy.deepcopy(ids)
        ids = np.asarray(ids)
        #mask_pos_okay = np.asarray(mask_pos_okay)

        #sz = len(ids)
        ## decide elements to mask
        #mask = np.full(sz, False)
        #num_mask = int(
        #    # add a random number for probabilistic rounding
        #    MASK_PROB * sz + np.random.random())

        #mask_idc = np.random.choice(self.max_range[mask_pos_okay],
        #                            num_mask,
        #                            replace=False)
        #mask[mask_idc] = True
        ## IMPORTANT!!! - start
        #masked = copy.deepcopy(mask).tolist()
        ## IMPORTANT!!! - end

        # debug - start - 2024-8-29
        tokenized_messages = ids.tolist()
        return tokenized_messages, gt, mask_pos_okay
        # debug - end

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        cls_label = LABEL2NUM[sample.pop(LABEL)]

        tokens, gt, masked = self.tokenize_messages(sample)

        gt.insert(0, SPECIAL_TOKENS[BOS])
        tokens.insert(0, SPECIAL_TOKENS[BOS])
        masked.insert(0, False)

        gt.append(SPECIAL_TOKENS[CLS])
        tokens.append(SPECIAL_TOKENS[CLS])
        masked.append(False)

        return {
            "tokens": tokens,
            "gt": gt,
            "mask": masked,
            "cls_label": cls_label,
        }
