"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random
from typing import List, Callable
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray

from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.triplethandler import TripletHandler

from typing import Protocol

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

    def get_used_triplets(self) -> []:
        return
