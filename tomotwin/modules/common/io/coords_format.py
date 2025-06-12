import pandas as pd
class CoordsFormat:

    @staticmethod
    def read(pth: str) -> pd.DataFrame:
        names = ["x", "y", "z"]
        return pd.read_csv(pth, delim_whitespace=True, header=None, index_col=False, dtype=float, names=names,
                           usecols=[0, 1, 2])

    def write(self, results: pd.DataFrame, path: str) -> None:
        results[["X", "Y", "Z"]].to_csv(path, index=False, header=None, sep=" ")

    def get_extension(self) -> str:
        return ".coords"