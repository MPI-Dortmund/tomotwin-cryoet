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
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import tomotwin
from tomotwin.modules.common.distances import DistanceManager
from tomotwin.modules.common.utils import check_for_updates
from tomotwin.modules.inference.argparse_map_ui import MapArgParseUI
from tomotwin.modules.inference.distance_mapper import DistanceMapper
from tomotwin.modules.inference.map_ui import MapMode, MapConfiguration
from tomotwin.modules.inference.mapper import Mapper
from tomotwin.modules.inference.reference_refinement import ReferenceRefiner


class FormatNotImplemented(Exception):
    '''
    Exception when an embedding format is not implemented.
    '''
    ...

def read_embeddings(path) -> pd.DataFrame:
    """
    Read the embeddings
    :return: DataFrame with embeddings
    """
    if path.endswith(".txt"):
        df = pd.read_csv(path)
    elif path.endswith((".pkl",".temb")):
        df = pd.read_pickle(path)
    else:
        raise FormatNotImplemented()
    dtypes = {}
    cols = df.columns
    for col_index, t in enumerate(df.dtypes):
        if is_numeric_dtype(t):
            dtypes[cols[col_index]] = np.float16
        else:
            dtypes[cols[col_index]] = t
    old_attrs = df.attrs
    casted = df.astype(dtypes)
    casted.attrs = old_attrs
    return casted


def run_map(
        mapper: Mapper, reference: np.array, volumes: np.array
) -> np.array:
    """
    Return the mapping of a references to a volume
    :param mapper: Mapper instance
    :param reference: Embeddings of the references
    :param volumes: Embeddings of a volume
    :return: Headmap for potential locations
    """
    return mapper.map(embeddings=volumes, references=reference)


def run(conf: MapConfiguration) -> None:
    '''
    Runs the map command
    :param conf: Configration from UI
    '''
    reference_embeddings_path = (
        conf.reference_embeddings_path
    )
    volume_embeddings_path = (
        conf.volume_embeddings_path
    )
    output_path = (
        conf.output_path
    )
    os.makedirs(output_path, exist_ok=True)

    if conf.mode == MapMode.DISTANCE:
        print("Read embeddings")
        reference_embeddings = read_embeddings(reference_embeddings_path)

        volume_embeddings = read_embeddings(volume_embeddings_path)

        volume_embeddings_np = volume_embeddings.drop(
            columns=["index", "filepath", "X", "Y", "Z"], errors="ignore"
        ).to_numpy()

        reference_embeddings_np = reference_embeddings.drop(
            columns=["index", "filepath", "X", "Y", "Z"], errors="ignore"
        ).to_numpy()


        dm = DistanceManager()
        distance = dm.get_distance(volume_embeddings.attrs["tomotwin_config"]["distance"])
        distance_func = distance.calc_np




        clf = DistanceMapper(distance_function=distance_func, similarty=distance.is_similarity())


        if not conf.skip_refinement:
            clf.quiet = True
            refiner = ReferenceRefiner(
                mapper=clf,
                sample_size=500
            )

            reference_embeddings_np = refiner.refine_references(references=reference_embeddings_np,embeddings=volume_embeddings_np, iterations=7)
            clf.quiet = False

        distances = run_map(
            mapper=clf,
            reference=reference_embeddings_np,
            volumes=volume_embeddings_np,
        )
        print("Prepare output...")
        del volume_embeddings_np
        del reference_embeddings_np

        ref_names = [
            os.path.basename(l) for l in reference_embeddings["filepath"]
        ]
        del reference_embeddings
        attributes = volume_embeddings.attrs
        df_data = {}
        #df_data["filename"] = volume_embeddings["filepath"].apply(lambda x: os.path.basename(x))
        if "X" in volume_embeddings:
            df_data["X"] = volume_embeddings["X"]
            df_data["Y"] = volume_embeddings["Y"]
            df_data["Z"] = volume_embeddings["Z"]
        del volume_embeddings

        for ref_index, _ in enumerate(ref_names):
            df_data[f"d_class_{ref_index}"] = distances[ref_index, :]


        classes_df = pd.DataFrame(
            df_data, copy=False
        )
        classes_df.attrs["tt_version_map"] = tomotwin.__version__

        # Add meta information from previous step
        for meta_key in attributes:
            classes_df.attrs[meta_key] = attributes[meta_key]

        # Add additional meta information
        classes_df.attrs["references"] = ref_names
        classes_df.attrs["skip_refinement"] = conf.skip_refinement
        classes_df.attrs["padding"] = volume_embeddings.attrs['padding']
        pth = os.path.join(output_path, "map.tmap")
        classes_df.to_pickle(pth)
        print(f"Wrote output to {pth}")



def _main_():
    ui = MapArgParseUI()
    ui.run()

    check_for_updates()

    conf = ui.get_map_configuration()
    run(conf=conf)


if __name__ == "__main__":
    _main_()