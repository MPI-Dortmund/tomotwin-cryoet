import pandas as pd
class CoordsFormat:

    def write(self, results: pd.DataFrame, path: str) -> None:
        results[["X", "Y", "Z"]].to_csv(path, index=False, header=None, sep=" ")

    def get_extension(self) -> str:
        return ".coords"