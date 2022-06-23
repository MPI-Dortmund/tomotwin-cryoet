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
