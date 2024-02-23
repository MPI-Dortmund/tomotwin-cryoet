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

import pandas as pd

from tomotwin.modules.training.filepathtriplet import FilePathTriplet


class TripletProvider(ABC):
    """
    Baseclass for creating triplets on a filename basis. It generates a List of FilePathTriplets
    """

    @abstractmethod
    def get_triplets(self) -> List[FilePathTriplet]:
        """
        :return: Returns a list of FileTriplet
        """

    @staticmethod
    def triplets_to_df(triplets: List[FilePathTriplet]) -> pd.DataFrame:
        """
        Convert the list of FileTriplets to a pandas dataframe
        :param triplets: List of FileTriplets
        :return: Pandas dataframe with FieTriplet infos.
        """
        df = pd.DataFrame(triplets)
        return df
