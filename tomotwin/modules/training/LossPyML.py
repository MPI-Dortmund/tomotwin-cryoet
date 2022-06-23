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

import torch.nn as nn
import torch
import numpy as np
from pytorch_metric_learning import miners, losses

class LossPyML(nn.Module):
    """
    Loss class for losses from the pytorch metric library.
    """

    def __init__(
        self,
        loss_func: losses.BaseMetricLossFunction,
        miner: miners.BaseTupleMiner = None
    ):
        super().__init__()

        self.miner = miner #miners.TripletMarginMiner(margin=self.margin,type_of_triplets="semihard")
        self.loss_func = loss_func #losses.TripletMarginLoss(margin=self.margin, distance=self.distance)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        :param anchor: Anchor tensor
        :param positive: Positive example tensor (same class as anchor)
        :param negative: Negative example tensor (different class as anchor)
        :return: Triplet loss
        """
        labels = []
        labels.extend(kwargs["label_anchor"][0])
        labels.extend(kwargs["label_positive"][0])
        labels.extend(kwargs["label_negative"][0])
        #Convert Labels
        unique_labels = np.unique(labels).tolist()
        labels_int = [unique_labels.index(l) for l in labels]
        labels = torch.tensor(labels_int)
        emb = torch.cat([anchor,positive,negative],dim=0) # concat all embeddings
        hard_pairs = None
        if self.miner:
            hard_pairs = self.miner(emb, labels)
        l = self.loss_func(emb, labels, hard_pairs)

        return l
