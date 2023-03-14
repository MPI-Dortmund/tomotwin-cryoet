Tutorial 2: Clustering based particle picking
============================================

We are working on a streamlined version of the clustering workflow. Will follow soon.


1. Downscale your Tomogram to 10 Å
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. include:: text_modules/downscale.rst

2. Embed your Tomogram
^^^^^^^^^^^^^^^^^^^^^^^
.. include:: text_modules/embed.rst

3. Estimate UMAP manifold
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $

    tomotwin_tools.py umap -i your_tomo_a10/embed/tomo/your_tomo_a10_embeddings.temb -o out/umap/

4. Find target clusters
^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash $

    tomotwin_tools.py embedding_mask -i out/embed/tomo/tiltseries_rec_embeddings.temb -o out/visualization/

.. prompt:: bash $

    napari your_tomo_a10.mrc out/visualization/your_tomo_a10_embeddings_label_mask.mrci

:guilabel:`Plugins` -> :guilabel:`napari-tomotwin` -> :guilabel:`napari-tomotwin` -> :guilabel:`Cluster UMAP embeddings`


5. Map your tomogram
^^^^^^^^^^^^^^^^^^^^

The map command will calculate the pairwise distances/similarity between the targets and the tomogram subvolumes and generate a localization map:

.. prompt:: bash $

    tomotwin_map.py distance -r your_tomo_a10/embed/targets/embeddings.temb -v your_tomo_a10/embed/tomo/your_tomo_a10_embeddings.temb -o your_tomo_a10/map/

6. Localize potential particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: text_modules/locate.rst


7. Inspect your particles with the boxmanager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Content comes here :-)

8. Scale your coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: text_modules/scale.rst