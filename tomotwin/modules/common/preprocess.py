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
import re

import numpy as np


def label_filename(path: str) -> str:
    filename = os.path.basename(path)
    try:
        lbl = re.search("id(?P<PDB>\d[a-zA-Z0-9]{3})", filename).group("PDB").upper()
    except AttributeError:
        lbl = re.search("\d[a-zA-Z0-9]{3}", filename).group(0).upper()  # https://regex101.com/r/rZi0TZ/1

    return lbl

def norm(vol: np.array) -> np.array:
    """Applies standard normalization"""
    return (vol-np.mean(vol))/np.std(vol)

def norm2(vol: np.array) -> np.array:
    min_val = np.min(vol)
    max_val = np.max(vol)
    normalized_vol = (vol - min_val) / (max_val - min_val)
    return normalized_vol