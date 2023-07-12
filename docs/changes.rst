Changes
=======

Version 0.5.0 (upcoming)
*************

* The ``tomotwin_embed.py tomogram`` command has now a optional ``--mask`` option to select region of interestes for embeddings.
* The ``tomotwin_tools.py embedding_mask`` now calculates a mask that masks out some portions of the tomogram volume that probably do not contain proteins. Using the generated mask when running ``tomotwin_embed.py tomogram``, the embeddings step is 2 times faster. CAUTION: In TomoTwin 0.4 the ``embeddings_mask`` command calculated a label mask for the clustering workflow. This functionality now happens automatically during the calculation of the umap (``tomotwin_tools.py umap``).
* For the clustering workflow, you can now calculate the medoid instead of arithmetic mean. This should be a much better representation of the cluster center.


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
* Added important tools like ``tomotwin_tools.py umap`` and ``tomotwin_tools.py embeddings_mask``
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