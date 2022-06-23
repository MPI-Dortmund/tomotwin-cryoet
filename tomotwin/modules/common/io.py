import mrcfile
import numpy as np


def read_mrc(pth: str) -> np.array:
    vol = mrcfile.open(pth, mode="r", permissive=True)
    return vol.data
