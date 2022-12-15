from typing import Protocol
import pandas as pd
import os

class CoordinateWriter(Protocol):

    def write(self, results: pd.DataFrame, path: os.PathLike) -> None:
      """Write coordinate results to disk"""

    def get_extension(self) -> str:
       "Return the extension of the format"
