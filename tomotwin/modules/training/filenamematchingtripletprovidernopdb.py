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

import os
from glob import glob
from typing import List

import numpy as np
import tqdm

from tomotwin.modules.common.preprocess import label_filename
from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.tripletprovider import TripletProvider


class FilenameMatchingTripletProviderNoPDB(TripletProvider):
    """
    Specific Implementation of the TripletProvider base class. Uses simple filename matching to create the FilePathTriplets.
    It is no using any PDB files. Instead it will generate all possible anchor-positive pairs and then select the negative randomly.
    """

    def __init__(
        self,
        path_volume: str,
        mask_volumes="*.mrc",
        shuffle: bool=True,
    ):
        """
        :param path_pdb: Path to folder with pdb files
        :param path_volume: Path to folder with volume files
        :param mask_volumes: Glob mask for volumes files
        :param max_neg: Maximum number of triplets generated for a anchor-positive pair.
        :param shuffle: If true, the list of volume files get shuffled.
        """
        self.path_volume = path_volume
        self.mask_volumes = mask_volumes
        self.shuffle = shuffle


    def get_triplets(self) -> List[FilePathTriplet]:
        volumefiles = glob(os.path.join(self.path_volume, self.mask_volumes), recursive=True)
        triplets = self.generate_triplets(volumefiles)
        np.random.shuffle(triplets)
        return triplets



    def generate_triplets(
        self, volumefiles: List[str]
    ) -> List[FilePathTriplet]:
        """
        :param volumefiles:  List of path to volume files
        :return: List of FileTriplets
        """

        def get_combinations(lbl_indices: np.array):
            shuffled_list = lbl_indices.copy()
            np.random.shuffle(shuffled_list)
            offset = 0
            if len(shuffled_list) % 2 != 0:
                offset = -1
            pairs = [ (shuffled_list[i], shuffled_list[i+1]) for i in range(0,len(shuffled_list)+offset, 2)]

            return pairs

        labels = [label_filename(p) for p in volumefiles]
        lbl_arr = np.array(labels)
        unique_labels = np.unique(lbl_arr)
        triplets = []
        for lbl in tqdm.tqdm(unique_labels,"Generate triplets (no pdb)"):
            index_same_lbl = np.where(lbl_arr == lbl)[0]
            index_different_lbl = np.where(lbl_arr != lbl)[0]
            if len(index_same_lbl) == 1:
                continue
            anchor_pos = get_combinations(index_same_lbl)
            #itertools.combinations(index_same_lbl,2)
            #anchor_pos = list(a)
            try:
                neg_index = np.random.choice(index_different_lbl,size=len(anchor_pos),replace=False)
            except ValueError:
                #sample with replacement
                neg_index = np.random.choice(index_different_lbl,size=len(anchor_pos),replace=True)

            for index, index_anc_pos in enumerate(anchor_pos):
                index_a, index_b = index_anc_pos
                trip = FilePathTriplet(volumefiles[index_a], volumefiles[index_b], volumefiles[neg_index[index]])
                triplets.append(trip)

        return triplets


