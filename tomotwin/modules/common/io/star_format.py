import pandas as pd
from pyStarDB import sp_pystardb as star

class StarFormat:

    def write(self, results: pd.DataFrame, path: str) -> None:
        columns = ['_rlnCoordinateX', '_rlnCoordinateY','_rlnCoordinateZ']
        df = pd.DataFrame(results[["X", "Y", "Z"]].to_numpy(), columns=columns)
        sfile = star.StarFile(path)
        sfile.update('', df, True)
        sfile.write_star_file(overwrite=True)

    def get_extension(self) -> str:
        return "_relion3.star"