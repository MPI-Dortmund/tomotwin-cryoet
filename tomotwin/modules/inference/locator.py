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

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import tqdm


class Locator(ABC):

    @abstractmethod
    def locate(self, map_output : pd.DataFrame) -> List[pd.DataFrame]:
        """
        Uses the tomotwin_map output to locate particles for each target reference
        """

    @staticmethod
    def extract_subclass_df(map: pd.DataFrame) -> List[pd.DataFrame]:
        '''
        Extract a dataframe for each reference
        :param map: Resullt from running map
        '''
        sub_dfs = []
        for i in range(len(map.attrs["references"])):
            sub = map[["X", "Y", "Z", f"d_class_{i}"]]
            sub.attrs["ref_name"] = map.attrs["references"][i]
            sub.attrs["ref_index"] = i
            sub_dfs.append(sub)
        return sub_dfs

    @staticmethod
    def nms(boxes: pd.DataFrame, size: int, nms_threshold=0.6) -> pd.DataFrame:
        # np.array(len(coords))
        distance_column = f"metric_best"
        boxes.sort_values(distance_column, inplace=True)
        boxes.reset_index(drop=True, inplace=True)
        boxes["width"] = size
        boxes["height"] = size
        boxes["depth"] = size
        boxes_data = np.empty(shape=(len(boxes), 7))
        kd_data = boxes[["X", "Y", "Z"]].to_numpy()
        from sklearn.neighbors import KDTree
        tree = KDTree(kd_data)

        for i in range(len(boxes)):
            row = boxes.loc[i]
            boxes_data[i, 0] = row["X"]
            boxes_data[i, 1] = row["Y"]
            boxes_data[i, 2] = row["Z"]
            boxes_data[i, 3] = row["width"]
            boxes_data[i, 4] = row["height"]
            boxes_data[i, 5] = row["depth"]
            boxes_data[i, 6] = 1  # merged

        for i in tqdm.tqdm(range(len(boxes_data)), desc="Non-maximum-supression"):
            box_i = boxes_data[i, :]

            close_indicis = tree.query_radius(kd_data[i:(i + 1), :], r=size)[0]
            if box_i[6] == 0:
                continue
            else:
                ones = np.ones((len(close_indicis), 7))
                boxes_i_rep = ones * box_i

                ious = Locator._bbox_iou_vec_3d(boxes_i_rep, boxes_data[close_indicis])
                iou_mask = np.empty(len(boxes_data), dtype=int)
                iou_mask_close = ious > nms_threshold

                iou_mask[close_indicis] = iou_mask_close
                iou_mask[i] = 0  # ignore current
                iou_mask = iou_mask == 1

                boxes_data[iou_mask, 6] = 0

        boxes = boxes[boxes_data[:, 6] == 1]
        boxes.reset_index(drop=True, inplace=True)
        return boxes

    @staticmethod
    def _bbox_iou_vec_3d(boxesA: np.array, boxesB: np.array) -> np.array:
        # 0 x
        # 1 y
        # 2 z
        # 3 w
        # 4 h
        # 5 depth

        x1_min = boxesA[:, 0] - boxesA[:, 3] / 2
        x1_max = boxesA[:, 0] + boxesA[:, 3] / 2
        y1_min = boxesA[:, 1] - boxesA[:, 4] / 2
        y1_max = boxesA[:, 1] + boxesA[:, 4] / 2
        z1_min = boxesA[:, 2] - boxesA[:, 5] / 2
        z1_max = boxesA[:, 2] + boxesA[:, 5] / 2

        x2_min = boxesB[:, 0] - boxesB[:, 3] / 2
        x2_max = boxesB[:, 0] + boxesB[:, 3] / 2
        y2_min = boxesB[:, 1] - boxesB[:, 4] / 2
        y2_max = boxesB[:, 1] + boxesB[:, 4] / 2
        z2_min = boxesB[:, 2] - boxesB[:, 5] / 2
        z2_max = boxesB[:, 2] + boxesB[:, 5] / 2

        intersect_w = Locator._interval_overlap_vec(x1_min, x1_max, x2_min, x2_max)
        intersect_h = Locator._interval_overlap_vec(y1_min, y1_max, y2_min, y2_max)
        intersect_depth = Locator._interval_overlap_vec(z1_min, z1_max, z2_min, z2_max)
        intersect = intersect_w * intersect_h * intersect_depth
        union = boxesA[:, 3] * boxesA[:, 4] * boxesA[:, 5] + boxesB[:, 3] * boxesB[:, 4] * boxesB[:,
                                                                                           5] - intersect
        return intersect / union

    @staticmethod
    def _interval_overlap_vec(x1_min, x1_max, x2_min, x2_max):
        intersect = np.zeros(shape=(len(x1_min)))
        cond_a = x2_min < x1_min
        cond_b = cond_a & (x2_max >= x1_min)
        intersect[cond_b] = np.minimum(x1_max[cond_b], x2_max[cond_b]) - x1_min[cond_b]
        cond_c = ~cond_a & (x1_max >= x2_min)
        intersect[cond_c] = np.minimum(x1_max[cond_c], x2_max[cond_c]) - x2_min[cond_c]

        return intersect