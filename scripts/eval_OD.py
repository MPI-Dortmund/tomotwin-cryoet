import argparse
import glob
import json
import os
import re
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate

from tomotwin.modules.common.preprocess import label_filename
from tomotwin.modules.inference.locator import Locator

def create_parser():
    parser = argparse.ArgumentParser()
    parser_parent.add_argument(
        "-p",
        "--positions",
        type=str,
        required=True,
        help="Ground truth positions of all particles"
    )

    parser_parent.add_argument(
        "-c",
        "--candidates",
        type=str,
        default=None,
        required=False,
        help="Candidate positions to evaluate"
    )

    parser_parent.add_argument(
        "--fbeta",
        type=int,
        default=6,
        help="Beta for F-Statistic"
    )

    parser_parent.add_argument(
        "--size",
        type=int,
        default=37,
        help="Boxsize for IOU calculation"
    )


    return parser


def get_fbeta(recall, precision, beta=1):
    if precision == 0 and recall == 0:
        return 0

    f_score = np.nan_to_num((1 + beta * beta) * precision * recall / (beta * beta * precision + recall))

    return f_score


def _add_size(df, size) -> pd.DataFrame:
    size = size
    df["width"] = size
    df["height"] = size
    df["depth"] = size

    return df


def locate_positions_stats(OD_results, class_positions, iou_thresh, fbeta):
    class_stats = {}
    locate_results_np =  OD_results[["X", "Y", "Z", "width", "height", "depth"]].to_numpy()
    true_positive = 0
    false_negative = 0
    found = np.array([False] * len(locate_results_np))
    for class_pos in class_positions.to_numpy():

        ones = np.ones((len(locate_results_np), 6))
        class_pos_rep = ones * class_pos
        ious = Locator._bbox_iou_vec_3d(class_pos_rep, locate_results_np)
        iou_mask = ious > iou_thresh

        if np.count_nonzero(iou_mask) >= 2:
            import inspect
            callerframerecord = inspect.stack()[1]
            frame = callerframerecord[0]
            info = inspect.getframeinfo(frame)
            print(f"{np.count_nonzero(iou_mask)} Maxima?? WAIT WHAT? oO")
        if np.any(iou_mask):

            found[np.argmax(ious)] = True

            true_positive = true_positive + 1
        else:
            false_negative = false_negative + 1
    false_positive = np.sum(np.array(found) == False)
    true_positive_rate = true_positive / len(class_positions)

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = get_fbeta(recall, precision, beta=fbeta)
    class_stats["Fbeta"] = float(f1_score)
    class_stats["Recall"] = recall
    class_stats["Precision"] = float(precision)
    class_stats["TruePositiveRate"] = float(true_positive_rate)
    class_stats["TP"] = int(true_positive)
    class_stats["FP"] = int(false_positive)
    class_stats["FN"] = int(false_negative)
    return class_stats

class LocateEvaluator():

    def __init__(self, positions_path: str, OD_results_path: str, fbeta: int, size: int):
        self.positions_path = positions_path
        self.OD_results_path = OD_results_path
        self.size = size
        self.fbeta = fbeta
        self.iou_thresh = 0.6


    def run(self) -> Dict:
        positions = pd.read_csv(self.positions_path, sep=",")
        positions.columns = ["class","X","Y","Z","rz","rx","ry"]
        positions = _add_size(positions, self.size)

        candidates = pd.read_csv(self.OD_results_path, sep=",")
        candidates.columns = ["X", "Y", "Z"]
        candidates = _add_size(candidates, self.size)



        stats = locate_positions_stats(locate_results, class_positions, self.iou_thresh, self.fbeta)

        for class_res_path in OD_results_paths:

            class_name = label_filename(os.path.basename(class_res_path))


            class_positions = positions[positions["class"]==class_name]
            class_positions = class_positions[["X", "Y", "Z", "width", "height", "depth"]]
            locate_results = pd.read_csv(class_res_path, sep=" ", header=None)
            locate_results["class"] = class_name
            locate_results.columns =[["X","Y","Z","class"]]

            locate_results = _add_size(locate_results, self.size, self.size_dict)



            stats[class_name] = class_stats

        return stats

    @staticmethod
    def print_stats(stats: dict, output_path: str=''):
        '''Prints the statistics as table'''

        np.set_printoptions(2)
        table = []
        header = ["#", "PDB"]

        for gt_class_index, gtclass in enumerate(stats):
            row = [gt_class_index + 1]
            row.append(gtclass)
            for stat_name in stats[gtclass]:
                if stat_name not in header:
                    header.append(stat_name)
                row.append(stats[gtclass][stat_name])
            table.append(row)

        row_means = ["AVG:", "-"]
        stats["AVG"] = {}
        for index, stat in enumerate(header):
            if stat != "PDB" and stat != "#":
                l = [row[index] for row in table]
                avg=np.nanmean(l)
                row_means.append(np.nanmean(l))
                stats["AVG"][stat] = float(avg)
        table.append(row_means)
        tab = tabulate(table, headers=header)
        print(tab)
        print("-------")
        tab_pth = os.path.join(output_path, 'evaluation_optim.txt')
        with open(tab_pth, 'w') as f:
            f.write(tab)
        print(f"Wrote evaluation table to {tab_pth}")
        stats_json = json.dumps(stats, indent=2)
        tab_json_pth = os.path.join(output_path, 'evaluation_optim_dict.json')
        with open(tab_json_pth, "w") as f:
            f.write(stats_json)
        print(f"Wrote evaluation table as json to {tab_json_pth}")


def readprobs(path):
    if path.endswith(".txt"):
        return pd.read_csv(path)
    elif path.endswith(".pkl"):
        return pd.read_pickle(path)
    else:
        print("Format not implemented")

def _main_():
    parser = create_parser()
    args = parser.parse_args()
    import sys
    positions_path = args.positions
    picks_path = args.candidates
    fbeta = args.fbeta
    size = args.size

    evaluator = LocateEvaluator(positions_path=positions_path, OD_results_path=picks_path, fbeta=fbeta, size=size)
    stats = evaluator.run()



if __name__ == "__main__":

    _main_()