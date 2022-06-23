from tomotwin.modules.inference.pick_ui import PickUI
from tomotwin.modules.inference.argparse_pick_ui import PickArgParseUI, PickConfiguration
import pandas as pd
import os
import numpy as np
from pyStarDB import sp_pystardb as star


def write_cbox(coordinates: pd.DataFrame, boxsize: int, path : str) -> None:
    '''
    Write results as CBOX to disk
    :param coordinates: Picking results
    :param boxsize: Box size that should be use
    :param path: Path of the new CBOX file
    '''

    def coords_to_np_array(xyz: pd.DataFrame, boxsize: int) -> np.array:
        num_boxes = len(xyz)
        num_fields = 11
        coords = np.zeros(shape=(num_boxes, num_fields))

        i=0
        for _, row in xyz.iterrows():
            c = float(row["metric_best"])
            coords[i, 0] = row["X"]-(boxsize/2)
            coords[i, 1] = row["Y"]-(boxsize/2)
            coords[i, 2] = row["Z"]
            coords[i, 3] = boxsize
            coords[i, 4] = boxsize
            coords[i, 5] = boxsize
            coords[i, 6] = float(row["size"])
            coords[i, 7] = float(row["size"])
            coords[i, 8] = c
            coords[i, 9] = None
            coords[i, 10] = None
            i=i+1


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


    coords = coords_to_np_array(coordinates, boxsize)

    include_slices = [a for a in np.unique(coords[:, 2]).tolist() if not np.isnan(a)]


    sfile = star.StarFile(path)

    version_df = pd.DataFrame([["1.0"]], columns=['_cbox_format_version'])
    sfile.update('global', version_df, False)

    df = pd.DataFrame(coords, columns=columns)
    sfile.update('cryolo', df, True)

    include_df = pd.DataFrame(include_slices, columns=['_slice_index'])
    sfile.update('cryolo_include', include_df, True)

    sfile.write_star_file(overwrite=True, tags=['global', 'cryolo', 'cryolo_include'])

def filter(locate_results : pd.DataFrame, conf: PickConfiguration ) -> pd.DataFrame:
    '''
    Applies several filter like best metric or min and max size.
    :param locate_results: Picking results
    :param conf: Configuration that contains all thresholds
    :return: Filtered picking results
    '''
    # Apply similarity threshold
    if conf.min_metric:
        locate_results = locate_results[locate_results["metric_best"] >= conf.min_metric]
    if conf.max_metric:
        locate_results = locate_results[locate_results["metric_best"] <= conf.max_metric]

    # Apply min max size filter
    if conf.min_size:
        locate_results = locate_results[locate_results["size"] >= conf.min_size]

    if conf.max_size:
        locate_results = locate_results[locate_results["size"] <= conf.max_size]
    return locate_results

def write_results(locate_results: pd.DataFrame, output_path: str, target: str) -> None:
    '''
    Write results to disk
    :param locate_results: Dataframe with picking results
    :param output_path: Path where to write results
    :param target: Target name
    '''
    os.makedirs(output_path,exist_ok=True)

    locate_results[["X", "Y", "Z"]].to_csv(os.path.join(output_path, f"{target}.coords"), index=False, header=None,
                                        sep=" ")
    if "width" not in locate_results:
        print("'width' column is missing in locate results. Use default box size of 37")
        size = 37
    else:
        size = np.unique(locate_results["width"])[0]
    write_cbox(locate_results, size, os.path.join(output_path, f"{target}.cbox"))


def run(ui: PickUI) -> None:
    '''
    Runs the picking pipeline
    :param ui: Settings from the UI
    '''
    # Load pickle
    ui.run()
    conf = ui.get_pick_configuration()

    # Get picks for target reference
    locate_results = pd.read_pickle(conf.locate_results_path)
    print("Found the following references:")
    referenes = np.unique(locate_results["predicted_class_name"])
    for ref in referenes:
        print(f"  - {ref}")

    if conf.target_reference is None:
        targets = np.unique(locate_results["predicted_class_name"])
    else:
        targets = conf.target_reference
    print("Filtering:")
    for target in targets:
        print(f"  - {target}")

    for target in targets:
        if target not in referenes:
            print(f"Target {target} is not a known reference. Skip")
            continue
        selection_target = target == locate_results["predicted_class_name"]
        locate_results_target = locate_results[selection_target]

        locate_results_target = filter(locate_results_target,conf)
        print(f"Target: {target} - Write {len(locate_results_target)} positions to disk.")
        write_results(locate_results_target,conf.output_path, target=target)


def _main_():
    ui = PickArgParseUI()
    run(ui=ui)



if __name__ == "__main__":
    _main_()