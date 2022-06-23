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

from glob import glob
from tomotwin.modules.training.tripletprovider import TripletProvider
from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from typing import List
import numpy as np
import os
import tqdm

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
