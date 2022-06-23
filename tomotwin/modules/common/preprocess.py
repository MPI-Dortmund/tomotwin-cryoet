import numpy as np
import os
import re

def label_filename(path: str) -> str:
    filename = os.path.basename(path)
    regex = "\d[a-zA-Z0-9]{3}"  # https://regex101.com/r/rZi0TZ/1
    return re.search(regex, filename).group(0).upper()

def norm(vol: np.array) -> np.array:
    """Applies standard normalization"""
    return (vol-np.mean(vol))/np.std(vol)
