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

import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import List, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd
from tqdm import tqdm

from tomotwin.modules.common.findmax.findmax import find_maxima
from tomotwin.modules.inference.locator import Locator


class FindMaximaLocator(Locator):
    def __init__(
        self,
        tolerance: float,
        stride: Tuple[int, int, int],
        window_size: int,
        global_min: float = 0.5,
        processes: int = None
    ):
        self.stride = stride
        self.window_size = window_size
        self.tolerance = tolerance
        self.min_size = None
        self.max_size = None
        self.output = None
        self.global_min = global_min
        self.processes = processes
        if self.processes is None:
            self.processes = multiprocessing.cpu_count()

    @staticmethod
    def to_volume(
        df: pd.DataFrame,
        target_class: int,
        stride : Tuple[int],
        window_size: int,
    ) -> Tuple[np.array, np.array]:
        # Convert to volume:
        half_bs_x = df["X"].min()
        half_bs_y = df["Y"].min()
        half_bs_z = df["Z"].min()
        # half_bs = (window_size - 1) / 2 This is modified because of padding
        x_val = (df["X"].values - half_bs_x) / stride[0]
        x_val = x_val.astype(int)
        y_val = (df["Y"].values - half_bs_y) / stride[1]
        y_val = y_val.astype(int)
        z_val = (df["Z"].values - half_bs_z) / stride[2]
        z_val = z_val.astype(int)


        # This array contains the distance(similarity)/probability at each coordinate
        vol = np.zeros(shape=(np.max(x_val) + 1, np.max(y_val) + 1, np.max(z_val) + 1))

        # Fill the array
        vals = df[f"d_class_{int(target_class)}"].values

        vol[(x_val, y_val, z_val)] = vals

        # Fill index array
        vol = vol.astype(np.float16)

        return vol

    @staticmethod
    def maxima_to_df(
        maximas: List[Tuple[float, float, float]],
        target: int,
        stride: Tuple[int],
        boxsize: int

    ) -> pd.DataFrame:

        bshalf = (boxsize-1)//2
        dat = {
            "X": [],
            "Y": [],
            "Z": [],
            "size": [],
            "metric_best": [],

        }
        for maxima, size, max_val in maximas:
            dat["size"].append(size)
            dat["metric_best"].append(max_val)
            dat["X"].append(maxima[0]*stride[0]+bshalf)
            dat["Y"].append(maxima[1]*stride[1]+bshalf)
            dat["Z"].append(maxima[2]*stride[2]+bshalf)


        dat = pd.DataFrame(dat)
        dat["predicted_class"]=target
        dat["X"] = dat["X"].astype(np.float16)
        dat["Y"] = dat["Y"].astype(np.float16)
        dat["Z"] = dat["Z"].astype(np.float16)
        dat["predicted_class"] = dat["predicted_class"].astype(np.int8)
        dat["size"] = dat["size"].astype(np.uint16)
        dat["metric_best"] = dat["metric_best"].astype(np.float16)
        return dat

    @staticmethod
    def apply_findmax_dask(vol: np.array,
                           tolerance: float,
                           global_min: float,
                           **kwargs
                           ) -> List[Tuple]:
        '''
        Applies the findmax procedure the 3d volume
        :param vol: Volume where maximas needs to be detected.
        :param tolerance: Prominence of the peak
        :param global_min: global minimum
        :param kwargs: kwargs arguments
        :return: List with 3 elements. First element is the maxima position, second element is the size (region growing), third element is maxima value
        '''

        da_vol = da.from_array(vol, chunks=200)  # really constant 200?
        lazy_results = []
        offsets = []
        indicis = list(itertools.product(*map(range, da_vol.blocks.shape)))
        with tqdm(total=len(indicis), position=kwargs.get("tqdm_pos"),
                  desc=f"Locate class {kwargs['tqdm_pos']}") as pbar:

            def find_max_bar_wrapper(*args, **kwargs):
                r = find_maxima(*args, **kwargs)
                pbar.update(1)
                return r

            for inds in indicis:
                chunk = da_vol.blocks[inds]
                offsets.append([a * b for a, b in zip(da_vol.chunksize, inds)])
                lr = dask.delayed(find_max_bar_wrapper)(np.asarray(chunk), tolerance=tolerance, global_min=global_min,
                                                        tqdm_pos=kwargs.get("tqdm_pos"), pbar=pbar)
                lazy_results.append(lr)

            # futures = dask.persist(*lazy_results)
            a = dask.compute(*lazy_results)
            # maximas, _

        # apply offsets
        maximas = []
        # k = 0
        for k, (maximas_in_chunk, _) in enumerate(a):
            off_chunk = offsets[k]
            for s_i, single_maxima in enumerate(maximas_in_chunk):
                new_pos = tuple([a + b for a, b in zip(single_maxima[0], off_chunk)])
                new_entry = [new_pos]
                new_entry.extend(single_maxima[1:])
                maximas_in_chunk[s_i] = new_entry
            maximas.extend(maximas_in_chunk)
            # k = k + 1

        return maximas

    @staticmethod
    def apply_findmax(vol: np.array,
                      tolerance: float,
                      global_min: float,
                      **kwargs
                      ) -> List[Tuple]:
        '''
        Applies the findmax procedure the 3d volume
        :param vol: Volume where maximas needs to be detected.
        :param tolerance: Prominence of the peak
        :param global_min: global minimum
        :param kwargs: kwargs arguments
        :return: List with 3 elements. First element is the maxima position, second element is the size (region growing), third element is maxima value
        '''

        maximas, _ = find_maxima(vol, tolerance, global_min=global_min, tqdm_pos=kwargs.get("tqdm_pos"))
        del _

        maximas = [
            m for m in maximas if m[1] > 1
        ]  # more than one pixel coordinate must be involved.

        return maximas

    @staticmethod
    def locate_class(class_id,
                     map_output: pd.DataFrame,
                     window_size: int,
                     stride: Tuple[int],
                     tolerance: float,
                     global_min: float,
                     ) -> Tuple[pd.DataFrame, np.array]:
        vol = FindMaximaLocator.to_volume(map_output, target_class=class_id, window_size=window_size, stride=stride)
        maximas = FindMaximaLocator.apply_findmax_dask(vol=vol,
                                                       class_id=class_id,
                                                       window_size=window_size,
                                                       stride=stride,
                                                       tolerance=tolerance,
                                                       global_min=global_min,
                                                       tqdm_pos=class_id)

        maximas = [
            m for m in maximas if m[1] > 1
        ]  # more than one pixel coordinate must be involved.

        print("done", class_id)
        particle_df = FindMaximaLocator.maxima_to_df(
            maximas, class_id, stride=stride, boxsize=window_size
        )

        return particle_df.copy(deep=True), vol

    def locate_(self, map_output: pd.DataFrame) -> pd.DataFrame:
        '''
        Run locate for a specific target
        :param map_output: Output dataframe from map command
        :return: dataframe with located positions
        '''

        print("start locate ", map_output.attrs['ref_name'])
        df_class, vol = FindMaximaLocator.locate_class(map_output.attrs['ref_index'], map_output, self.window_size,
                                                       self.stride, self.tolerance, self.global_min)
        df_class.attrs["name"] = map_output.attrs['ref_name']
        df_class.attrs["heatmap"] = vol
        print("Located", df_class.attrs["name"], len(df_class))
        return df_class

    def locate(self, map_output: pd.DataFrame) -> List[pd.DataFrame]:
        '''
        starts the locate process in parallel
        :param map_output: Output of the map command
        :return: List of dataframes. One for each target.
        '''
        sub_dfs = Locator.extract_subclass_df(map_output)
        with Pool(self.processes) as pool:
            class_frames_and_vols = list(pool.map(self.locate_, sub_dfs))

        return class_frames_and_vols
