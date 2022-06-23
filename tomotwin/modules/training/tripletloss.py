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
from tomotwin.modules.common.distances import Distance

class TripletLoss(nn.Module):
    """
    Implementation of the triplet loss
    """

    def __init__(
        self,
        margin: float,
        distance: Distance,
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        :param anchor: Anchor tensor
        :param positive: Positive example tensor (same class as anchor)
        :param negative: Negative example tensor (different class as anchor)
        :return: Triplet loss
        """
        distance_positive = self.distance.calc(anchor, positive)
        distance_negative = self.distance.calc(anchor, negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)
        l = losses.mean()

        return l, distance_positive, distance_negative
