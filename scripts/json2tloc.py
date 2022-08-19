import json
import numpy as np
import csv
import pandas as pd
import os
import sys
import argparse




def make_results(boxes, box_size, class_id):
    dat = np.zeros(shape=(len(boxes), 3))
    dat_tloc = {}
    dat_tloc["X"] = []
    dat_tloc["Y"] = []
    dat_tloc["Z"] = []
    dat_tloc["metric_best"] = []
    dat_tloc["predicted_class"] = []
    dat_tloc["width"] = []
    dat_tloc["height"] = []
    dat_tloc["depth"] = []
    dat_tloc["size"] = []
    for box_i, box in enumerate(boxes):
        dat[box_i, 0] = box[0]
        dat[box_i, 1] = box[1]
        dat[box_i, 2] = box[2]

        dat_tloc["X"].append(box[0])
        dat_tloc["Y"].append(box[1])
        dat_tloc["Z"].append(box[2])
        dat_tloc["metric_best"].append(box[4])
        dat_tloc["predicted_class"].append(class_id)
        dat_tloc["width"].append(box_size)
        dat_tloc["height"].append(box_size)
        dat_tloc["depth"].append(box_size)
        dat_tloc["size"].append(box[5])

    return dat, pd.DataFrame(dat_tloc)

def _main_():

    parser = argparse.ArgumentParser(description="Converts eman2 info json files (from template matching) to tloc coordinates")
    parser.add_argument('--json', nargs='+', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    pths = args.json
    out_pth_base = args.out


    for pth in pths:
        json_filename = os.path.splitext(os.path.basename(pth))[0]
        out_pth = os.path.join(out_pth_base, json_filename)

        os.makedirs(out_pth, exist_ok=True)
        with open(pth) as json_file:
            data = json.load(json_file)

        boxes = data['boxes_3d']
        for box in boxes:
            if not float(box[6]).is_integer():
                print("JSON format changed. Stop")
                # if that happen, I need to adapt the script to the current master version ofeman
                import sys
                sys.exit(1)

        class_list = data['class_list']
        refs = []

        tloc_data = []
        coords_data = []

        for class_id in class_list:
            cid = int(class_id)
            class_name = class_list[class_id]["name"]
            refs.append(class_name)
            box_size = class_list[class_id]["boxsize"]
            class_boxes = [box for box in boxes if box[6] == cid] # in eman2 master it is different!
            dat, dat_tloc = make_results(class_boxes, box_size, cid)
            tloc_data.append(dat_tloc)
            coords_data.append((dat, class_name))

        os.makedirs(out_pth, exist_ok=True)

        df = pd.concat(tloc_data)
        df.attrs["references"] = refs

        df.to_pickle(os.path.join(out_pth, f"tomo.tloc"))
        '''
        for boxes, class_name in coords_data:

            with open(os.path.join(out_pth, f"{class_name}.coords"), "w") as boxfile:
                boxwriter = csv.writer(
                    boxfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_NONE
                )

                for box in boxes:
                    # box.x / box.y = Lower left corner
                    boxwriter.writerow([box[0], box[1], box[2]])
        '''

if __name__ == '__main__':
    _main_()