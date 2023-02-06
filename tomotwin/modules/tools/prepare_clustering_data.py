import argparse
import os
from tqdm import tqdm
import cuml
import numpy as np
import pandas as pd
import mrcfile
from numpy.typing import ArrayLike

parser = argparse.ArgumentParser(description="Calculates umap embeddings and a segmentation mask for the lasso  tool")
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Embeddings file to use for clustering')

parser.add_argument('-o', '--output', type=str, required=True,
                    help='Output folder')
parser.add_argument('--fit_sample_size', type=int, default=400000,
                    help='Sample size using for the fit of the umap')

parser.add_argument('--chunk_size', type=int, default=500000,
                    help='Chunk size for transform all data')
args = parser.parse_args()


def calculate_umap(
        embeddings : pd.DataFrame,
        fit_sample_size: int,
        transform_chunk_size: int) -> ArrayLike:
    print("Prepare data")
    embeddings = pd.read_pickle(embeddings)
    embeddings = embeddings.reset_index(drop=True)
    fit_sample = embeddings.sample(n=min(len(embeddings) ,fit_sample_size), random_state=17)
    fit_sample = fit_sample.drop(['filepath', 'Z', 'Y', 'X'], axis=1, errors='ignore')
    all_data = embeddings.drop(['filepath', 'Z', 'Y', 'X'] ,axis=1, errors='ignore')
    reducer = cuml.UMAP(
        n_neighbors=200,
        n_components=2,
        n_epochs=None,  # means automatic selection
        min_dist=0.0,
        random_state=19
    )
    print(f"Fit umap on {len(fit_sample)} samples")
    reducer.fit(fit_sample)

    num_chunks = max(1, int(len(all_data) / transform_chunk_size))
    print(f"Transform complete dataset in {num_chunks} chunks with a chunksize of ~{int(len(all_data ) /num_chunks)}")

    chunk_embeddings = []
    for chunk in tqdm(np.array_split(all_data, num_chunks) ,desc="Transform"):
        embedding = reducer.transform(chunk)
        chunk_embeddings.append(embedding)

    embedding = np.concatenate(chunk_embeddings)
    all_data['umap_0'] = embedding[:, 0]
    all_data['umap_1'] = embedding[:, 1]
    umap_labels = all_data[['umap_0', 'umap_1']].copy()

    return umap_labels

def create_segmentation_map(embeddings):
    print("Create segmentation map")
    embeddings = pd.read_pickle(embeddings)
    Z = embeddings.attrs['tomogram_input_shape'][0]
    Y = embeddings.attrs['tomogram_input_shape'][1]
    X = embeddings.attrs['tomogram_input_shape'][2]
    embeddings = embeddings.reset_index(drop=True)
    segmentation_mask = embeddings[['Z', 'Y', 'X']].copy()
    segmentation_mask = segmentation_mask.reset_index()
    empty_array = np.zeros(shape=(Z, Y, X))
    for row in segmentation_mask.itertuples(index=True, name='Pandas'):
        X = int(row.X)
        Y = int(row.Y)
        Z = int(row.Z)
        label = int(row.index)
        empty_array[Z, Y, X] = label + 1
    segmentation_array = empty_array.astype(np.float32)

    return segmentation_array


def _main_():
    out_pth = args.output
    umap_embeddings = calculate_umap(embeddings=args.input, fit_sample_size=args.fit_sample_size, transform_chunk_size=args.chunk_size)
    os.makedirs(out_pth, exist_ok=True)
    print("Write umap embeddings to disk")
    umap_embeddings.to_csv(
        os.path.join(out_pth, os.path.splitext(os.path.basename(args.input))[0] + "_umap.csv"))
    segmentation_layer = create_segmentation_map(embeddings=args.input)
    print("Write segmentation mask to disk")
    with mrcfile.new(os.path.join(out_pth, "embedding_mask.mrci"), overwrite=True) as mrc:
        mrc.set_data(segmentation_layer)



if __name__ == '__main__':
    _main_()