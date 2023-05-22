Changes
=======

Version 0.4.3
*************

* Fix numba related issue by updating various dependencies (https://github.com/MPI-Dortmund/tomotwin-cryoet/issues/20):
    - Python 3.9 -> Python 3.10
    - Rapids 22.04 -> 23.04
    - CUDA 11.6 -> CUDA 11.8
* Updating pyStarDB from 0.3.2 -> 0.4.2

Version 0.4.0
*************

* Official clustering workflow release. Please checkout the updated installation instructions and in depth tutorial.
* Added important tools like ``tomotwin_tools umap`` and ``tomotwin_tools embeddings_mask``
* Added more unit tests

Version 0.3.0
*************

* Scale heatmaps to the same size as the tomogram, to make them overlay easily. Additionally, make them optional (--write_heatmaps) as they require some space
* Write Relion 3 STAR files as they are required for WARP (Thanks Tom Dendooven)
* Reference refinement deactivated by default, as we noticed that it makes the results worse in some cases.

Version 0.2.1
*************

* Training crashed because the package name was outdated in the training module.

Version 0.2.0
*************

* Initial release