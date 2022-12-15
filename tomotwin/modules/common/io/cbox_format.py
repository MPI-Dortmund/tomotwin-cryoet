import pandas as pd
import numpy as np
from pyStarDB import sp_pystardb as star

class CBoxFormat:



    def write(self, results: pd.DataFrame, path: str) -> None:
        def coords_to_np_array(xyz: pd.DataFrame, boxsize: int) -> np.array:
            num_boxes = len(xyz)
            num_fields = 11
            coords = np.zeros(shape=(num_boxes, num_fields))

            i = 0
            for _, row in xyz.iterrows():
                c = float(row["metric_best"])
                coords[i, 0] = row["X"] - (boxsize / 2)
                coords[i, 1] = row["Y"] - (boxsize / 2)
                coords[i, 2] = row["Z"]
                coords[i, 3] = boxsize
                coords[i, 4] = boxsize
                coords[i, 5] = boxsize
                coords[i, 6] = float(row["size"])
                coords[i, 7] = float(row["size"])
                coords[i, 8] = c
                coords[i, 9] = None
                coords[i, 10] = None
                i = i + 1

            return coords

        columns = []
        columns.append('_CoordinateX')
        columns.append('_CoordinateY')
        columns.append('_CoordinateZ')
        columns.append('_Width')
        columns.append('_Height')
        columns.append('_Depth')
        columns.append('_EstWidth')
        columns.append('_EstHeight')
        columns.append('_Confidence')
        columns.append('_NumBoxes')
        columns.append('_Angle')

        if "width" not in results:
            print("'width' column is missing in locate results. Use default box size of 37")
            boxsize = 37
        else:
            boxsize = np.unique(results["width"])[0]

        coords = coords_to_np_array(results, boxsize)

        include_slices = [a for a in np.unique(coords[:, 2]).tolist() if not np.isnan(a)]

        sfile = star.StarFile(path)

        version_df = pd.DataFrame([["1.0"]], columns=['_cbox_format_version'])
        sfile.update('global', version_df, False)

        df = pd.DataFrame(coords, columns=columns)
        sfile.update('cryolo', df, True)

        include_df = pd.DataFrame(include_slices, columns=['_slice_index'])
        sfile.update('cryolo_include', include_df, True)

        sfile.write_star_file(overwrite=True, tags=['global', 'cryolo', 'cryolo_include'])

    def get_extension(self) -> str:
        return ".cbox"