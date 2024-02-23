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

from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.tripletprovider import TripletProvider


class FilenameMatchingTripletProvider(TripletProvider):
    """
    Specific Implementation of the TripletProvider base class. Uses simple filename matching to create the FilePathTriplets
    """

    def __init__(
        self,
        path_pdb: str,
        path_volume: str,
        mask_pdb="*.mrc",
        mask_volumes="*.mrc",
        max_neg: int=1,
        shuffle: bool=True,
    ):
        """
        :param path_pdb: Path to folder with pdb files
        :param path_volume: Path to folder with volume files
        :param mask_pdb: Glob mask for pdb files.
        :param mask_volumes: Glob mask for volumes files
        :param max_neg: Maximum number of triplets generated for a anchor-positive pair.
        :param shuffle: If true, the list of volume files get shuffled.
        """
        self.path_pdb = path_pdb
        self.path_volume = path_volume
        self.mask_pdb = mask_pdb
        self.mask_volumes = mask_volumes
        self.max_neg = max_neg
        self.shuffle = shuffle

    def generate_triplets(
        self, pdbfiles: List[str], volumefiles: List[str]
    ) -> List[FilePathTriplet]:
        """
        This procedure assumes they the PDB ID is the filename of pdbfile. If that is not the case
        this procecure will not work.
        :param pdbfiles: List of path to pdb files
        :param volumefiles:  List of path to volume files
        :return: List of FileTriplets
        """

        triplets = []
        if self.shuffle:
            np.random.shuffle(volumefiles)

        neg_candidates = volumefiles.copy()
        volumefiles_filenames = [os.path.splitext(os.path.basename(v))[0].upper() for v in volumefiles]
        neg_choosen = []
        no_match = 0
        for pdbfile in tqdm.tqdm(pdbfiles, desc="Generating triplets"):
            pdb = os.path.splitext(os.path.basename(pdbfile))[0].upper()
            for vol_index, positive_volume_candidate in enumerate(volumefiles):

                if pdb in volumefiles_filenames[vol_index]:

                    chosen =0
                    selections = []
                    while chosen<self.max_neg:
                        cand_index = np.random.randint(0,len(neg_candidates))
                        cand = neg_candidates[cand_index]
                        if pdb not in cand.upper():
                            selections.append(cand)
                            neg_candidates.remove(cand)
                            chosen = chosen +1
                            if len(neg_candidates)<10:
                                neg_candidates = volumefiles.copy()
                            no_match = 0
                        else:
                            no_match = no_match + 1
                            if no_match == 10:
                                neg_candidates = volumefiles.copy()
                                no_match=0

                    new_triplets = [
                        FilePathTriplet(pdbfile, positive_volume_candidate, neg)
                        for neg in selections
                    ]

                    neg_choosen.extend(selections)
                    triplets.extend(new_triplets)

        return triplets

    def get_triplets(self) -> List[FilePathTriplet]:

        pdbfiles = glob(os.path.join(self.path_pdb, self.mask_pdb), recursive=True)
        volumefiles = glob(os.path.join(self.path_volume, self.mask_volumes), recursive=True)
        triplets = self.generate_triplets(pdbfiles, volumefiles)
        np.random.shuffle(triplets)
        return triplets
