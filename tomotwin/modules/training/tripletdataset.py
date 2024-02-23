"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""

import random
from typing import List, Callable
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.triplethandler import TripletHandler

LabelExtractorFunc = Callable[[str], str]

class Transformer(Protocol):

    def transform(self, data: NDArray) -> NDArray:
        ...

class TripletDataset(Dataset):
    """
    Implementation of Torch Dataset. It uses the triplethandler to provide the triplets as its needed by pytorch.
    """

    def __init__(
        self,
        training_data: List[FilePathTriplet],
        handler: TripletHandler,
        label_ext_func: LabelExtractorFunc,
        augmentation_volumes: Transformer = None,
        augmentation_anchors: Transformer = None,
    ):
        self.training_data = training_data
        random.shuffle(self.training_data)
        self.augmentation_volumes = augmentation_volumes
        self.augmentation_anchors = augmentation_anchors
        self.handler = handler
        self.label_ext_func = label_ext_func

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, item_index):

        triplet = self.training_data[item_index]
        nptriplet = self.handler.handle(triplet)
        anchor_vol = nptriplet.anchor
        if self.augmentation_anchors:
            anchor_vol = self.augmentation_anchors.transform(anchor_vol)
        anchor_vol = anchor_vol.astype(np.float32)
        anchor_vol = anchor_vol[np.newaxis]
        anchor_vol = torch.from_numpy(anchor_vol)

        positive_vol = nptriplet.positive
        if self.augmentation_volumes:
            positive_vol = self.augmentation_volumes.transform(positive_vol)
        positive_vol = positive_vol.astype(np.float32)
        positive_vol = positive_vol[np.newaxis]
        positive_vol = torch.from_numpy(positive_vol)

        negative_vol = nptriplet.negative
        if self.augmentation_volumes:
            negative_vol = self.augmentation_volumes.transform(negative_vol)
        negative_vol = negative_vol.astype(np.float32)
        negative_vol = negative_vol[np.newaxis]
        negative_vol = torch.from_numpy(negative_vol)

        lbl_neg = self.label_ext_func(triplet.negative)
        lbl_pos = self.label_ext_func(triplet.positive)

        input_triplet = {
            "anchor": anchor_vol,
            "positive": positive_vol,
            "negative": negative_vol,
            "filenames": [triplet.anchor, triplet.positive, triplet.negative],
            "label_anchor": [self.label_ext_func(triplet.anchor)],
            "label_positive": [lbl_pos],
            "label_negative": [lbl_neg],
        }

        return input_triplet

    def get_triplet_dimension(self):

        nptriplet = self.handler.handle(self.training_data[0])
        assert nptriplet.anchor.shape == nptriplet.positive.shape
        assert nptriplet.negative.shape == nptriplet.negative.shape

        return nptriplet.anchor.shape

    def get_used_triplets(self) -> []:
        return
