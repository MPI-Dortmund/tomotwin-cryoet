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

    def to_volume(
        self, df: pd.DataFrame, target_class: int, use_p: bool = False
    ) -> Tuple[np.array, np.array]:
        # Convert to volume:
        half_bs = (self.window_size - 1) / 2
        x_val = (df["X"].values - half_bs) / self.stride[0]
        x_val = x_val.astype(int)
        y_val = (df["Y"].values - half_bs) / self.stride[1]
        y_val = y_val.astype(int)
        z_val = (df["Z"].values - half_bs) / self.stride[2]
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

    def maxima_to_df(
        self,
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

    def locate(self, classify_output: pd.DataFrame) -> List[pd.DataFrame]:

        particles_dataframes = []

        df_particles = classify_output
        unique_classes = classify_output.attrs["references"]

        for id,name in enumerate(unique_classes):

            if id < 0:
                continue

            vol, index_vol = self.to_volume(df_particles, target_class=id)
            #volp, _ = self.to_volume(df_particles, target_class=id, use_p=True)

            maximas, mask = find_maxima(vol, self.tolerance, global_min=self.global_min)
            self.unfiltered = maximas
            maximas_filtered = [
                m for m in maximas if m[1] > 1
            ]  # more than one pixel coordinate must be involved.
            particle_df = self.maxima_to_df(
                maximas_filtered, df_particles, index_vol, id, name
            )
            particle_df.attrs["name"] = name
            particles_dataframes.append(particle_df.copy(deep=True))

            if self.output is not None:
                print("Write", name, len(particle_df))
                with mrcfile.new(
                    os.path.join(self.output, name + ".mrc"), overwrite=True
                ) as mrc:
                    vol = vol.astype(np.float32)
                    vol = vol.swapaxes(0, 2)
                    mrc.set_data(vol)

        return particles_dataframes
