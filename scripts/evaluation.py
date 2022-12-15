import argparse
import numpy as np
from typing import Dict, List, Callable
import pandas as pd
from tabulate import tabulate
import os
import re
import glob
import tqdm
from tomotwin.modules.inference.locator import Locator
from tomotwin.modules.common.preprocess import label_filename
import json

def create_subvolume_parser(parser_parent):

    parser_parent.add_argument(
        "-p",
        "--probabilties",
        type=str,
        required=True,
        help="Path to probabilities file"
    )

    parser_parent.add_argument(
        "-r",
        "--reference",
        type=str,
        required=True,
        help="Path to reference embeddings file"
    )
    return parser_parent

def create_position_parser(parser_parent):
    parser_parent.add_argument(
        "-p",
        "--positions",
        type=str,
        required=True,
        help="particle_positions.txt of a SHREC Tomogram"
    )

    parser_parent.add_argument(
        "-l",
        "--locate",
        type=str,
        required=True,
        help="Path to locate results"
    )

    parser_parent.add_argument(
        "-s",
        "--size",
        type=str,
        default=None,
        required=False,
        help="Path to json file with box sizes for each reference"
    )

    parser_parent.add_argument(
        "--optim",
        action='store_true',
        default=False,
        help="If given, it optimized the parameters"
    )

    parser_parent.add_argument(
        "--stepsize_optim_similarity",
        type=float,
        default=0.05,
    )



def create_parser():
    """Create evaluation parser"""
    parser_parent = argparse.ArgumentParser(
        description="Evaluation script for TomoTwin (SHREC)"
    )
    subparsers = parser_parent.add_subparsers(help="sub-command help")

    subvolume_parser = subparsers.add_parser(
        "subvolumes",
        help="Statistics with ground truth subvolumes as input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    locate_parser = subparsers.add_parser(
        "positions",
        help="Statistics for location script with ground truth positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    create_subvolume_parser(subvolume_parser)
    create_position_parser(locate_parser)

    return parser_parent

def get_size_dict_json(pth: str) -> dict:
    import json
    with open(pth) as f:
        data = json.load(f)
        data = {key.upper(): value for key, value in data.items()}
    return data

def get_size_dict() -> dict:

    return {
        "4V94": 37,
        "4CR2": 33,
        "1QVR": 25,
        "1BXN": 19,
        "3CF3": 25,
        "1U6G": 18,
        "3D2F": 22,
        "2CG9": 18,
        "3H84": 18,
        "3GL1": 13,
        "3QM1": 12,
        "1S3X": 12,
        "5MRC": 37,
        "1FPY": 18,
        "1FO4": 23,
        "1JZ8": 19,
        "1CU1": 17,
        "1SS8": 17,
        "6AHU": 18,
        "6TPS": 28,
        "6X9Q": 37,
        "6GY6": 33,
        "6NI9": 12,
        "6VZ8": 30,
        "4HHB": 12,
        "7B7U": 20,
        "VESICLE": None,
        "FIDUCIAL": 18,
    }

def _add_size(df, size, size_dict = None) -> pd.DataFrame:
    if size_dict is None:
        size = size
        df["width"] = size
        df["height"] = size
        df["depth"] = size
    else:
        df["width"] = 0
        df["height"] = 0
        df["depth"] = 0
        for row_index, row in df.iterrows():
            size = size_dict[str(row["class"]).upper()]
            df.at[row_index,"width"] = size
            df.at[row_index, "height"] = size
            df.at[row_index, "depth"] = size

    return df

def locate_positions_stats(locate_results, class_positions, iou_thresh):
    class_stats = {}
    locate_results_np =  locate_results[["X", "Y", "Z", "width", "height", "depth"]].to_numpy()
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
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    class_stats["F1"] = float(f1_score)
    class_stats["Recall"] = recall
    class_stats["Precision"] = float(precision)
    class_stats["TruePositiveRate"] = float(true_positive_rate)
    class_stats["TP"] = int(true_positive)
    class_stats["FP"] = int(false_positive)
    class_stats["FN"] = int(false_negative)
    return class_stats

class SubvolumeEvaluator():

    def __init__(self):
        pass

    @staticmethod
    def stats_class_wise(ground_truth: List[int],
                         predicted: List[int],
                         ) -> Dict[int,Dict[str,float]]:

        gt_classes = np.unique(ground_truth)
        stats = {}
        for gtclass in gt_classes:
            stats[gtclass] = {}
            mask_predicted = predicted==gtclass
            mask_gt = ground_truth==gtclass
            true_positives = np.sum(np.logical_and(mask_gt,mask_predicted))
            true_negatives = np.sum(np.logical_and(np.logical_not(mask_gt),np.logical_not(mask_predicted)))
            false_positives = np.sum(np.logical_and(np.logical_not(mask_gt),mask_predicted))
            false_negative = np.sum(np.logical_and(mask_gt,np.logical_not(mask_predicted)))

            true_negative_rate = true_negatives/np.sum(mask_gt==False)
            true_positive_rate = true_positives /np.sum(mask_gt)

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives/(true_positives+false_negative)

            balanced_accuracy = (true_positive_rate+true_negative_rate)/2
            f1_score = 2*precision*recall/(precision+recall)
            stats[gtclass]["F1"] = float(f1_score)
            stats[gtclass]["Recall"] = float(recall)
            stats[gtclass]["Precision"] = float(precision)
            stats[gtclass]["TrueNegativeRate"] = float(true_negative_rate)
            stats[gtclass]["TruePositiveRate"] = float(true_positive_rate)
            stats[gtclass]["BalancedAccuracy"] = float(balanced_accuracy)
        return stats

    @staticmethod
    def evaluate(probabilties: pd.DataFrame,
                 pdb_id_converter : Dict,
                 filename_pdb_extactor: Callable[[str], str]
                 ) -> Dict[int,Dict[str,float]]:
        '''
        Calculates the statistics. Returns a dictonary with statistics for each class (identified by an integer).
        Each class statistics dictionary consists of a string identified as key and the statistics value (float).
        '''


        filenames_volumes = probabilties["filename"].tolist()

        volume_ground_truth_classes = []
        for name in filenames_volumes:
            pdb_str = filename_pdb_extactor(os.path.splitext(os.path.basename(name))[0]).upper()
            try:
                vclass = pdb_id_converter[pdb_str]
            except KeyError:
                print(f"No reference for class {pdb_str} not found. Skip it")
            volume_ground_truth_classes.append(vclass)
        stats = SubvolumeEvaluator.stats_class_wise(ground_truth=volume_ground_truth_classes,predicted=probabilties["predicted_class"])
        stats["VOID"] = np.sum(probabilties["predicted_class"]==-1)
        return stats

    @staticmethod
    def extract_pdb_from_filename(filename):
        regex = "\d[a-zA-Z0-9]{3}" # https://regex101.com/r/rZi0TZ/1
        return re.search(regex, filename).group(0)

    @staticmethod
    def print_stats(stats, pdb_id_converter, output_path: str=''):
        '''Prints the statistics as table'''

        np.set_printoptions(2)
        table = []
        header = ["#","PDB"]

        for gt_class_index, gtclass in enumerate(stats):
            if gtclass == "VOID":
                continue
            row = [gt_class_index+1]
            row.append(pdb_id_converter[gtclass])
            for stat_name in stats[gtclass]:
                if stat_name not in header:
                    header.append(stat_name)
                row.append(stats[gtclass][stat_name])
            table.append(row)

        row_means = ["AVG:", "-"]
        for index, stat in enumerate(header):
            if stat != "PDB" and stat != "#":
                l = [row[index] for row in table]
                row_means.append(np.nanmean(l))
        table.append(row_means)
        tab = tabulate(table,headers=header)
        print(tab)
        print("-------")
        print("Number of subvolumes in void class:", stats["VOID"])

        with open(os.path.join(output_path,'evaluation_optim.txt'), 'w') as f:
            f.write(tab)
            f.write("\n-------")
            f.write(f"\nNumber of subvolumes in void class: {stats['VOID']}")


    @staticmethod
    def shrec_pdb_class_id_converter(reference_embeddings: List[str]):
        '''
        Create an dictionary to convert class IDs to PDB strings and vice versa.
        '''

        pdb_class_dict = {}
        for class_num, name in enumerate(reference_embeddings['filepath']):
            pdb_str = os.path.splitext(os.path.basename(name))[0]
            pdb_str = pdb_str.upper()
            pdb_class_dict[class_num] = pdb_str
            pdb_class_dict[pdb_str] = class_num
        pdb_class_dict["0XXX"] = -1
        pdb_class_dict[-1] = "0XXX"

        return pdb_class_dict

class LocateOptimEvaluator():

    def __init__(self, positions_path: str, locate_results_path: str, sizes_pth: str, stepsize_optim_similarity: float = 0.05, size=37):
        self.positions_path = positions_path
        self.locate_results_path = locate_results_path
        self.size = size
        self.iou_thresh = 0.6
        self.size_dict= None
        if sizes_pth:
            self.size_dict = get_size_dict_json(sizes_pth)
        self.stepsize_optim_similarity = stepsize_optim_similarity


    def filter(self, df, min_val=None, max_val=None, field=None):

        dfc = df.copy()


        if field == None:
            return dfc

        if min_val != None:
            dfc = dfc[dfc[field] > min_val]

        if max_val != None:
            dfc = dfc[dfc[field] < max_val]

        return dfc

    def get_stats(self, df, positions):
        #label_filename()

        refs = df.attrs['references']
        pc = df['predicted_class'].iloc[0]
        class_name = os.path.splitext(refs[pc])[0]
        class_name = label_filename(class_name)

        pos_classes = np.array([cl.upper() for cl in positions["class"]])
        class_positions = positions[pos_classes == class_name.upper()]
        class_positions = class_positions[["X", "Y", "Z", "width", "height", "depth"]]
        # locate_results.columns = [["X", "Y", "Z", "class"]]

        df = df.rename(columns={"predicted_class_name": "class"})
        df["class"] = class_name
        df = _add_size(df, self.size, self.size_dict)

        stats = locate_positions_stats(locate_results=df, class_positions=class_positions, iou_thresh=self.iou_thresh)
        return stats

    def optim(self,locate_results, positions):

        def find_best(locate_results, field, range, stepsize, type):

            best_stats = self.get_stats(locate_results, positions)
            #print(best_stats)
            #import sys
            #sys.exit()
            best_f1 = best_stats["F1"]
            best_value = 0
            best_df = locate_results

            for val in np.arange(start=range[0], stop=range[1], step=stepsize):
                if type == "min":
                    df = self.filter(locate_results, min_val=val, field=field)
                if type == "max":
                    df = self.filter(locate_results, max_val=val, field=field)
                if len(df) == 0:
                    continue
                stats = self.get_stats(df, positions)
                if stats["F1"] > best_f1:
                    best_f1 = stats["F1"]
                    best_stats = stats
                    best_value = val
                    best_df = df.copy()
            return best_stats, best_df, best_value

        min_size_range = [1, 500]
        max_size_range = [1, 500]
        dsize = 2
        min_similarity_range = [0,1]
        dsim = self.stepsize_optim_similarity
        locate_results_id = locate_results
        o_dict = {}
        stats, locate_results_filtered, best_value = find_best(
            locate_results=locate_results_id,
            field="metric_best",
            range=min_similarity_range,
            stepsize=dsim,
            type="min"
        )
        if locate_results_filtered is not None:
            o_dict["O_METRIC"] = float(best_value)
            locate_results_id = locate_results_filtered

        stats, locate_results_filtered, best_value = find_best(
            locate_results=locate_results_id,
            field="size",
            range=min_size_range,
            stepsize=dsize,
            type="min"
        )
        if locate_results_filtered is not None:
            o_dict["O_MIN_SIZE"] = int(best_value)
            locate_results_id = locate_results_filtered

        stats, locate_results_filtered, best_value = find_best(
            locate_results=locate_results_id,
            field="size",
            range=max_size_range,
            stepsize=dsize,
            type="max"
        )
        if locate_results_filtered is not None:
            o_dict["O_MAX_SIZE"] = int(best_value)
            locate_results_id = locate_results_filtered

        stats.update(o_dict)

        return stats

    def run(self) -> Dict:
        stats = {}

        positions = pd.read_csv(self.positions_path, delim_whitespace=True, header=None)

        if len(positions.columns)==1:
            print("Read position with ',' sperator")
            positions = pd.read_csv(self.positions_path, sep=",")

        locate_results = pd.read_pickle(self.locate_results_path)
        unique_class_labels = np.sort(np.unique(locate_results['predicted_class']))

        if len(unique_class_labels) == 1 and len(positions.columns) == 3:
            # Single class run...
            print("Single class run!")
            positions['class'] = "0000"
            locate_results.attrs['references'] = ["0000"]
            positions.columns = ['X','Y',"Z",'class']
        else:
            positions.columns = ["class", "X", "Y", "Z", "rz", "rx", "ry"]
        gt_data_classes = np.unique(positions["class"])
        gt_data_classes = [gt.upper() for gt in gt_data_classes]

        positions = _add_size(positions, self.size, self.size_dict)


        for id in tqdm.tqdm(unique_class_labels,desc="Optimize"):
            dfc = locate_results[locate_results["predicted_class"] == id]
            try:
                reference_name = locate_results.attrs['references'][id]
                class_name = label_filename(reference_name)
            except AttributeError:
                print("Skip. Not valid class:", locate_results.attrs['references'][id])
                continue
            print(class_name)
            if class_name.upper() not in gt_data_classes:
                print("Skip ", class_name)
                print(f"{class_name} is not in ground truth positions data. Skip it.")
                continue

            best_stats = self.optim(dfc, positions) #self.get_stats(df, positions)
            stats[reference_name] = best_stats
        return stats

class LocateEvaluator():

    def __init__(self, positions_path: str, locate_results_path: str):
        self.positions_path = positions_path
        self.locate_results_path = locate_results_path
        self.size = 37
        self.iou_thresh = 0.6
        self.size_dict = get_size_dict()


    def run(self) -> Dict:
        positions = pd.read_csv(self.positions_path, sep=" ")
        positions.columns = ["class","X","Y","Z","rz","rx","ry"]

        positions = _add_size(positions, self.size, self.size_dict)

        locate_results_paths = glob.glob(os.path.join(self.locate_results_path,"*.coords"))
        stats = {}
        for class_res_path in locate_results_paths:

            class_name = label_filename(os.path.basename(class_res_path))

            #if class_name != "1QVR":
            #    continue

            class_positions = positions[positions["class"]==class_name]
            class_positions = class_positions[["X", "Y", "Z", "width", "height", "depth"]]
            locate_results = pd.read_csv(class_res_path, sep=" ", header=None)
            locate_results["class"] = class_name
            #print(f"CLASS {class_name} N (GT): {len(class_positions)} N (LOCATE): {len(locate_results)}")
            locate_results.columns =[["X","Y","Z","class"]]

            locate_results = _add_size(locate_results, self.size, self.size_dict)
            class_stats = locate_positions_stats(locate_results, class_positions, self.iou_thresh)


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
    if "subvolumes" in sys.argv[1]:
        probabilties_path = args.probabilties
        references_path = args.reference
        probs = readprobs(probabilties_path)
        references = readprobs(references_path)

        pdb_id_converter = SubvolumeEvaluator.shrec_pdb_class_id_converter(
            references,
        )

        stats = SubvolumeEvaluator.evaluate(probs, pdb_id_converter, filename_pdb_extactor=SubvolumeEvaluator.extract_pdb_from_filename)
        SubvolumeEvaluator.print_stats(stats, pdb_id_converter,output_path=os.path.dirname(probabilties_path))

    if "positions" in sys.argv[1]:
        positions_path = args.positions
        locate_path = args.locate
        optim = args.optim
        sizes_pth = args.size
        size = 37
        if args.size and args.size.isdigit():
            sizes_pth = None
            size = int(args.size)

        if optim:
            evaluator = LocateOptimEvaluator(positions_path=positions_path,
                                             locate_results_path=locate_path,
                                             sizes_pth=sizes_pth,
                                             stepsize_optim_similarity=args.stepsize_optim_similarity,
                                             size=size)
        else:
            evaluator = LocateEvaluator(positions_path=positions_path, locate_results_path=locate_path)
        stats = evaluator.run()

        LocateEvaluator.print_stats(stats, output_path=os.path.dirname(locate_path))






if __name__ == "__main__":

    _main_()