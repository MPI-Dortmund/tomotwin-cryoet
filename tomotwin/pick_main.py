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
import typing

import pandas as pd

from tomotwin.modules.common.io.coords_format import CoordsFormat
from tomotwin.modules.common.io.reader_writer import CoordinateWriter
from tomotwin.modules.common.io.star_format import StarFormat
from tomotwin.modules.common.utils import check_for_updates
from tomotwin.modules.inference.argparse_pick_ui import (
    PickArgParseUI,
    PickConfiguration,
)


class InvalidLocateResults(Exception):
    """
    Custrom exception for invalid locate results file
    """


def write_coords(results: pd.DataFrame, filepath: str) -> None:
    """
    Write picking results in .coords format.

    :param results: Picking results
    :param filepath: Filepath where the coords file is written to
    :return: None
    """
    results[["X", "Y", "Z"]].to_csv(filepath, index=False, header=None, sep=" ")


def filter_results(
    locate_results: pd.DataFrame, conf: PickConfiguration
) -> pd.DataFrame:
    """
    Applies several filter like best metric or min and max size.
    :param locate_results: Picking results
    :param conf: Configuration that contains all thresholds
    :return: Filtered picking results
    """
    # Apply similarity threshold
    if conf.min_metric:
        locate_results = locate_results[
            locate_results["metric_best"] >= conf.min_metric
        ]
    if conf.max_metric:
        locate_results = locate_results[
            locate_results["metric_best"] <= conf.max_metric
        ]

    # Apply min max size filter
    if conf.min_size:
        locate_results = locate_results[locate_results["size"] >= conf.min_size]

    if conf.max_size:
        locate_results = locate_results[locate_results["size"] <= conf.max_size]
    return locate_results


def write_results(
    locate_results: pd.DataFrame,
    writer: typing.List[CoordinateWriter],
    output_path: str,
    target: str,
) -> None:
    """
    Write results to disk
    :param locate_results: Dataframe with picking results
    :param output_path: Path where to write results
    :param target: Target name
    """

    if len(locate_results) == 0:
        raise InvalidLocateResults("Locate results are empty")
    os.makedirs(output_path, exist_ok=True)

    for w in writer:
        p = os.path.join(
            output_path, f"{os.path.splitext(target)[0]}{w.get_extension()}"
        )
        w.write(
            locate_results,
            p,
        )
        print(p)


def run(conf: PickConfiguration) -> None:
    """
    Runs the picking pipeline
    :param ui: Settings from the UI
    """
    # Get picks for target reference
    locate_results = pd.read_pickle(conf.locate_results_path)
    print()
    print("Found the following references:")
    references = locate_results.attrs["references"]
    for ref in references:
        print(f"  - {ref}")

    if conf.target_reference is None:
        targets = references
    else:
        targets = conf.target_reference
    print("Filtering:")
    for target in targets:
        print(f"  - {target}")

    # Setup writer
    writer: typing.List[CoordinateWriter] = []
    writer.append(StarFormat())
    writer.append(CoordsFormat())

    for target in targets:
        if target not in references:
            print(f"Target {target} is not a known reference. Skip")
            continue

        selection_target = references.index(target) == locate_results["predicted_class"]
        locate_results_target = locate_results[selection_target]

        locate_results_target = filter_results(locate_results_target, conf)
        print(
            f"Target: {target} - Write {len(locate_results_target)} positions to disk."
        )
        try:
            write_results(
                locate_results=locate_results_target,
                writer=writer,
                output_path=conf.output_path,
                target=target,
            )
        except InvalidLocateResults:
            print("Skip.")


def _main_():
    ui = PickArgParseUI()

    ui.run()
    conf = ui.get_pick_configuration()

    check_for_updates()

    run(conf=conf)


if __name__ == "__main__":
    _main_()
