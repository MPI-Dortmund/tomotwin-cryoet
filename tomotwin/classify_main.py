"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pandas as pd
import numpy as np
from tomotwin.modules.inference.classifier import Classifier
from tomotwin.modules.inference.distance_classifier import DistanceClassifier
from tomotwin.modules.inference.classify_ui import ClassifyUI, ClassifyMode
from tomotwin.modules.inference.argparse_classifiy_ui import ClassifiyArgParseUI
from tomotwin.modules.common.distances import DistanceManager

import os
from pandas.api.types import is_numeric_dtype


def read_embeddings(path):
    if path.endswith(".txt"):
        df = pd.read_csv(path)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        print("Format not implemented")
        return None
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


def classify(
    classifier: Classifier, reference: np.array, volumes: np.array
) -> np.array:
    return classifier.classify(embeddings=volumes, references=reference)


def run(ui: ClassifyUI):
    ui.run()
    conf = ui.get_classification_configuration()
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

    if conf.mode == ClassifyMode.DISTANCE:
        print("Read embeddings")
        reference_embeddings = read_embeddings(reference_embeddings_path)
        volume_embeddings = read_embeddings(volume_embeddings_path)

        print("Reading Done")
        volume_embeddings_np = volume_embeddings.drop(
            columns=["index", "filepath", "X", "Y", "Z"], errors="ignore"
        ).to_numpy()

        reference_embeddings_np = reference_embeddings.drop(
            columns=["index", "filepath", "X", "Y", "Z"], errors="ignore"
        ).to_numpy()

        dm = DistanceManager()
        distance = dm.get_distance(volume_embeddings.attrs["tomotwin_config"]["distance"])
        distance_func = distance.calc_np

        clf = DistanceClassifier(distance_function=distance_func, similarty=distance.is_similarity())
        _ = classify(
            classifier=clf,
            reference=reference_embeddings_np,
            volumes=volume_embeddings_np,
        )
        reference_embeddings_np = None
        volume_embeddings_np = None
        if distance.is_similarity():
            distances = clf.get_distances()
            classes = np.argmax(distances, axis=0)
            #class_dist = np.max(distances, axis=0)
            #classes[class_dist < threshold] = -1
        else:
            distances = clf.get_distances()
            classes = np.argmin(distances, axis=0)
            #class_dist = np.min(distances, axis=0)
            #classes[class_dist > threshold] = -1


        ref_names = [
            os.path.basename(l) for l in reference_embeddings["filepath"].tolist()
        ]
        vol_names = [
            os.path.basename(l) for l in volume_embeddings["filepath"].tolist()
        ]
        class_names = ["void"]*len(classes)
        for i, cl in enumerate(classes):
            if cl!=-1:
                class_names[i] = ref_names[cl]

        columns_data = []
        columnes_header = []

        if "X" in volume_embeddings:
            columns_data.append(volume_embeddings["X"])
            columnes_header.append("X")
            columns_data.append(volume_embeddings["Y"])
            columnes_header.append("Y")
            columns_data.append(volume_embeddings["Z"])
            columnes_header.append("Z")
        columns_data.append(vol_names)
        columns_data.append(classes)
        columns_data.append(class_names)
        columnes_header.append("filename")
        columnes_header.append("predicted_class")
        columnes_header.append("predicted_class_name")

        for ref_index, _ in enumerate(ref_names):
            columns_data.append(distances[ref_index, :])
            columnes_header.append(f"d_class_{ref_index}")

        classes_df = pd.DataFrame(
            data=list(zip(*columns_data)),
            columns=columnes_header,
        )

        # Add meta information from previous step
        for meta_key in volume_embeddings.attrs:
            classes_df.attrs[meta_key] = volume_embeddings.attrs[meta_key]

        # Add additional meta information
        classes_df.attrs["references"] = ref_names
        classes_df.to_pickle(os.path.join(output_path, "predicted.pkl"))



def _main_():
    ui = ClassifiyArgParseUI()
    run(ui=ui)


if __name__ == "__main__":
    _main_()
