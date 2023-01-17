import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter import filedialog
import os
import argparse


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, full_data: np.array, alpha_other=0.01):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.full_data = full_data
        self.indicis_full_data = None

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def get_indicis_full_data(self):
        return self.indicis_full_data

    def onselect(self, verts):
        path = Path(verts)

        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.indicis_full_data = np.nonzero(path.contains_points(self.full_data))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        #self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.fc[self.ind, 0] = 0.941
        self.fc[self.ind, 1] = 0.670
        self.fc[self.ind, 2] = 0.219

        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert clusters to coords', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-u','--umap', type=str, required=True,
                        help='Path to pickled pandas dataframe with the umap')
    parser.add_argument('-e','--embedding', type=str, required=True,
                        help='Path to tomogram embedding')
    parser.add_argument('-p', '--pointsize', type=float, default=0.001,
                        help='Path to tomogram embedding')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='Output directory')
    return parser

def _main_():
    parser = get_parser()
    args = parser.parse_args()
    pth_umap = args.umap
    pth_embedding = args.embedding
    emb_umap = pd.read_pickle(pth_umap)
    emb_data = pd.read_pickle(pth_embedding)
    references = []
    references_names = []
    if args.outdir is None:
        root = Tk()
        root.update()
        out_dir = filedialog.askdirectory(mustexist=False)
        root.destroy()
    else:
        out_dir = args.outdir
    os.makedirs(out_dir,exist_ok=True)

    print(f"Length umap", len(emb_umap))
    print(f"Length data", len(emb_data))

    emb_umap_selection = emb_umap.sample(n=min(len(emb_umap),500000))

    data = emb_umap_selection.to_numpy()
    minx = np.min(data[:, 0])
    maxx = np.max(data[:, 0])
    miny = np.min(data[:, 1])
    maxy = np.max(data[:, 1])
    subplot_kw = dict(xlim=(minx, maxx), ylim=(miny, maxy), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=args.pointsize)
    selector = SelectFromCollection(ax, pts,full_data=emb_umap.to_numpy())


    def accept(event):
        if "enter" in event.key:
            indicies = selector.get_indicis_full_data().tolist()
            emb_data_selection = emb_data.iloc[indicies]
            emb_data_selection = emb_data_selection.drop(columns=["X", "Y", "Z","filepath"], errors="ignore")

            new_ref = emb_data_selection.astype(np.float32).mean(axis=0)
            new_ref = new_ref.to_frame().T#reshape(32,1)

            references.append(new_ref)
            conc_ref = pd.concat(references, ignore_index=True)
            pth = os.path.join(out_dir,f"cluster_{len(references)}.temb")

            references_names.append(os.path.basename(pth))
            conc_ref["filepath"] = references_names
            plt.savefig(os.path.splitext(pth)[0]+".png")
            emb_data_selection.to_pickle(pth)
            print(f"Written: {pth}")
            pth_ref = os.path.join(out_dir, f"references.temb")
            conc_ref.to_pickle(pth_ref)
            print(f"Update: {pth_ref}")
            selector.disconnect()
            #ax.set_title("")
            fig.canvas.draw()


    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    _main_()



