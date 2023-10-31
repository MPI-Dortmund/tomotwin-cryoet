Changes
=======


Version 0.7.0
*************

* Fixes a bug in ``tomotwin_embed.py`` which leads to crashes because of duplicate embeddings
* Adds the experimental flag `--cosine` to the command ``tomotwin_tools.py umap`` which in theory should give better umaps (but its also slower).
* Adds the experimental floag ``--lower`` and ``--concat`` to the command ``tomotwin_tools.py filter_embedding``. With the new flags you can select embeddings which are at a certain distance to one or multiple references. This is handy if you want to fine tune your manuel selected reference targets.

Version 0.6.1
*************

Version 0.6.0
*************

* ``tomotwim_embed.py`` is now 1.6x faster and linearly scales across multiple GPUs
   * Exploiting new the ``compile`` option of the latest pytorch 2.1 nightly build.
   * Internally ``DistributedDataParallel`` is used instead of ``DataParallel``

Version 0.5.1
*************

* Unpin pytorch-metric-library in the requirements.

Version 0.5.0
*************

* Speed up embedding using Masks
    * The command ``tomotwin_embed.py tomogram`` now has an optional ``--mask`` argument to select the region of interest for embedding.
    * The command ``tomotwin_tools.py embedding_mask`` now computes a by isonet inspired mask that hides some parts of the tomogram volume that are unlikely to contain proteins. If you use the generated mask with new ``--mask`` argument, the embedding step is up to 2 times faster. **CAUTION:** In TomoTwin 0.4, the ``embeddings_mask`` command calculated a label mask for the clustering workflow. This functionality now happens automatically during the calculation of the umap (``tomotwin_tools.py umap``).
    * Thanks Caitie McCafferty and Ricardo Righetto for the feature request
* More accurate cluster centers
    * When selecting clusters in Napari during the clustering workflow, the `Medoid <https://en.wikipedia.org/wiki/Medoid>`_ is now calculated instead of the average of all cluster embeddings. This has the advantage that it is guaranteed to be on the embedding hypersphere and should be a better representation of the cluster center than the average.
    * The coordinates of the found medoid for each cluster is written as a .coords file to disk.
* Other
    * Updated installation instructions for napari: Napari 0.4.17 -> Napari 0.4.18
    * :ref:`Added some information snippets for developers <dev-info>`

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
