import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import mrcfile

from tomotwin.modules.inference.locator import Locator
from tomotwin.modules.common.findmax.findmax import find_maxima


class FindMaximaLocator(Locator):
    def __init__(
        self,
        tolerance: float,
        stride: Tuple[int, int, int],
        window_size: int,
        global_min: float = 0.5,
    ):
        self.stride = stride
        self.window_size = window_size
        self.tolerance = tolerance
        self.min_size = None
        self.max_size = None
        self.output = None
        self.global_min = global_min

    @staticmethod
    def to_volume(
        df: pd.DataFrame,
        target_class: int,
        stride : Tuple[int],
        window_size: int,
        use_p: bool = False
    ) -> Tuple[np.array, np.array]:
        # Convert to volume:
        half_bs = (window_size - 1) / 2
        x_val = (df["X"].values - half_bs) / stride[0]
        x_val = x_val.astype(int)
        y_val = (df["Y"].values - half_bs) / stride[1]
        y_val = y_val.astype(int)
        z_val = (df["Z"].values - half_bs) / stride[2]
        z_val = z_val.astype(int)

        # This array contains the distance(similarity)/probability at each coordinate
        vol = np.zeros(shape=(np.max(x_val) + 1, np.max(y_val) + 1, np.max(z_val) + 1))
        # This volumes contains the corresponding row index in the input data frame for each coordinate
        index_vol = np.zeros(
            shape=(np.max(x_val) + 1, np.max(y_val) + 1, np.max(z_val) + 1), dtype=int
        )

        # Fill the array
        if use_p:
            vals = df[f"p_class_{int(target_class)}"].values
        else:
            vals = df[f"d_class_{int(target_class)}"].values
        vol[(x_val, y_val, z_val)] = vals

        # Fill index array
        index_vol[(x_val, y_val, z_val)] = np.arange(len(vals))

        return vol, index_vol

    @staticmethod
    def maxima_to_df(
        maximas: List[Tuple[float, float, float]],
        df: pd.DataFrame,
        index_vol: np.array,
        target: int,
        class_name: str,
    ) -> pd.DataFrame:
        df = df.copy()
        selected_rows = []
        sizes = []
        region_best = []
        for maxima, size, max_val in maximas:

            try:
                row_index = index_vol[
                    int(np.round(maxima[0])),
                    int(np.round(maxima[1])),
                    int(np.round(maxima[2])),
                ]
                # if df.iloc[row_index]["predicted_class"] == target:
                selected_rows.append(row_index)
                sizes.append(size)
                region_best.append(max_val)
            except IndexError:
                print(
                    "Index error for",
                    maxima,
                    (
                        int(np.round(maxima[0])),
                        int(np.round(maxima[1])),
                        int(np.round(maxima[2])),
                    ),
                )

        selected_df = df.iloc[selected_rows].copy()
        selected_df["size"] = sizes
        selected_df["metric_best"] = region_best
        selected_df["predicted_class"] = target
        selected_df["predicted_class_name"] = class_name
        selected_df = selected_df[
            [
                "X",
                "Y",
                "Z",
                "filename",
                "predicted_class",
                "predicted_class_name",
                "size",
                "metric_best",
            ]
        ]
        return selected_df

    @staticmethod
    def apply_findmax(classify_output: pd.DataFrame,
                      class_id: int,
                      class_name: str,
                      window_size: int,
                      stride: Tuple[int],
                      tolerance: float,
                      global_min: float,

                      ) -> (pd.DataFrame, np.array):
        df_particles = classify_output
        vol, index_vol = FindMaximaLocator.to_volume(df_particles, target_class=class_id, window_size=window_size, stride=stride)

        maximas, mask = find_maxima(vol, tolerance, global_min=global_min)

        maximas_filtered = [
            m for m in maximas if m[1] > 1
        ]  # more than one pixel coordinate must be involved.

        particle_df = FindMaximaLocator.maxima_to_df(
            maximas_filtered, df_particles, index_vol, class_id, class_name
        )
        particle_df.attrs["name"] = class_name

        return particle_df, vol



    @staticmethod
    def locate_class(class_id,
                     class_name,
                     df_particles:
                     pd.DataFrame,
                     window_size: int,
                     stride: Tuple[int],
                     tolerance: float,
                     global_min: float,
                     output: str) -> pd.DataFrame:
        particle_df, vol = FindMaximaLocator.apply_findmax(classify_output=df_particles,
                                                           class_id=class_id,
                                                           class_name=class_name,
                                                           window_size=window_size,
                                                           stride=stride,
                                                           tolerance=tolerance,
                                                           global_min=global_min)

        if output is not None:
            with mrcfile.new(
                    os.path.join(output, class_name + ".mrc"), overwrite=True
            ) as mrc:
                vol = vol.astype(np.float32)
                vol = vol.swapaxes(0, 2)
                mrc.set_data(vol)
        print("Located", class_name, len(particle_df))
        return particle_df.copy(deep=True)




    def locate(self, classify_output: pd.DataFrame) -> List[pd.DataFrame]:

        particles_dataframes = []

        df_particles = classify_output
        unique_classes = classify_output.attrs["references"]

        from concurrent.futures import ProcessPoolExecutor as Pool
        from itertools import repeat

        with Pool() as pool:

            df_classes = pool.map(
                FindMaximaLocator.locate_class,
                list(range(len(unique_classes))),
                unique_classes,
                repeat(df_particles),
                repeat(self.window_size),
                repeat(self.stride),
                repeat(self.tolerance),
                repeat(self.global_min),
                repeat(self.output)
            )

        for df in df_classes:
            particles_dataframes.append(df)

        return particles_dataframes
