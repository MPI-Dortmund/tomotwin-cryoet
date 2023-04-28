Tutorial 2: Clustering based particle picking
============================================

We are working on a streamlined version of the clustering workflow. Will follow soon.


1. Downscale your Tomogram to 10 Å
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. include:: text_modules/downscale.rst

2. Embed your Tomogram
^^^^^^^^^^^^^^^^^^^^^^^
.. include:: text_modules/embed.rst

3. Estimate UMAP manifold and Generate Embedding Mask
^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we will approximate the tomogram embeddings to 2D to allow for efficient visualization. To calculate a UMAP
.. prompt:: bash $

    tomotwin_tools.py umap -i your_tomo_a10/embed/tomo/tomo_embeddings.temb -o your_tomo_a10/clustering/

Additionally, we will generate a mask of the embeddings to allow us to track which UMAP values correspond to which points in the tomogram. To generate this mask:
.. prompt:: bash $

    tomotwin_tools.py embedding_mask -i your_tomo_a10/embed/tomo/tomo_embeddings.temb -o your_tomo_a10/clustering/

4. Load data for clustering in Napari
^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have all the input files for the clustering workflow we can get started in Napari. First open your tomogram and the embedding mask
.. prompt:: bash $

    napari_boxmanager your_tomo_a10.mrc your_tomo_a10/clustering/your_tomo_a10_embedding_label_mask.mrci

Next open the napari-tomotwin clustering tool via

:guilabel:`Plugins` -> :guilabel:`napari-tomotwin` -> :guilabel:`Cluster UMAP embeddings`

Choose the :guilabel:`Path to UMAP` by clicking on :guilabel:`Select file` and provide the path to your your_tomo_a10_embeddings.tumap.

Click :guilabel:`Load` and after a second, a 2D plot of the umap embeddings should appear in the plugin window.

4. Find target clusters
^^^^^^^^^^^^^^^^^^^^^^^^

Outline a set of points in the 2D plot and these points will become highlighted in your tomogram.

Alternatively you can click in the tomogram and a small red circle appears around the embedding for this position in the tomogram.

You can use the :guilabel:`Magnifying glass` icon to change the displayed area/zoom and the :guilabel:`Home` icon to reset it.

In some cases, using a log scale for the histogram may show clusters that are difficult to distinguish. To activate the log scale click on :guilabel:`Advanced settings` :guilabel:`Log scale`.

You can select multiple targets at once by holding shift when outlining points.

Note: when generating targets to pick large proteins, it is best to outline points that only lay in the center of your protein rather than covering the entire protein. This will help ensure that your resulting picks are centered.

6. Save target clusters
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have outlined a target cluster for each protein of interest, it is time to save these targets to be used as picking references in this and additional tomograms.

This can be done with :guilabel:`Plugins` -> :guilabel:`napari-tomotwin` -> :guilabel:`Save cluster targets` and providing an output directory where cluster_targets.temb will be written.


7. Map your tomogram
^^^^^^^^^^^^^^^^^^^^

The map command will calculate the pairwise distances/similarity between the targets and the tomogram subvolumes and generate a localization map:

.. prompt:: bash $

    tomotwin_map.py distance -r your_tomo_a10/clustering/cluster_targets.temb -v your_tomo_a10/embed/tomo/your_tomo_a10_embeddings.temb -o your_tomo_a10/map/

6. Localize potential particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: text_modules/locate.rst


7. Inspect your particles with the boxmanager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Content comes here :-)

8. Scale your coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: text_modules/scale.rst