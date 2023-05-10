Changes
=======

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