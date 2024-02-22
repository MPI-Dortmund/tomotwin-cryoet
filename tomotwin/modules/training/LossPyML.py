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

import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import miners, losses


class LossPyML(nn.Module):
    """
    Loss class for losses from the pytorch metric library.
    """

    def __init__(
        self,
        loss_func: losses.BaseMetricLossFunction,
        miner: miners.BaseTupleMiner = None,
        only_negative_labels = None
    ):
        super().__init__()

        self.miner = miner #miners.TripletMarginMiner(margin=self.margin,type_of_triplets="semihard")
        self.loss_func = loss_func #losses.TripletMarginLoss(margin=self.margin, distance=self.distance)
        if only_negative_labels is None:
            self.only_negative_labels = []
        else:
            self.only_negative_labels = only_negative_labels

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

            ## Remove only negative labels
            only_negative_labels_asint = []
            for l in self.only_negative_labels:
                try:
                    uniq_index = unique_labels.index(l)
                    only_negative_labels_asint.append(uniq_index)
                except ValueError:
                    pass

            valid_indicies = [i for i in range(len(hard_pairs[0])) if labels_int[hard_pairs[0][i]] not in only_negative_labels_asint]
            try:
                new_hard_pairs = (hard_pairs[0][valid_indicies], hard_pairs[1][valid_indicies], hard_pairs[2][valid_indicies])
                hard_pairs = new_hard_pairs
            except RuntimeError as e:
                print("Runtime error. Use old hard pairs")
                print(valid_indicies)
                print(e)




        l = self.loss_func(emb, labels, hard_pairs)

        return l
