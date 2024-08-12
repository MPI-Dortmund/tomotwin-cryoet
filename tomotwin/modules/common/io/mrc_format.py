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

import mrcfile
import numpy as np

class MrcFormat:
    @staticmethod
    def read(pth: str) -> np.array:
        vol = None
        try:
            with mrcfile.open(pth, permissive=True) as mrc:
                vol = mrc.data.astype(np.float32)
        except ValueError as e:
            raise Exception(f"Failed reading {pth}") from e
        return vol
