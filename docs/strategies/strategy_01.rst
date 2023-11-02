Strategy 1: Refinement of references/targets using umaps
========================================================

When to use it
--------------

You have selected references or clusters targets but you are not happy with the picking results. The embedding that is calculated based on a cluster or a reference is not always a ideal representation. Some references simply don't work well and sometimes umap does not show all the structure that is actually in the umap embedding.

What it does
------------

This stragey takes your references/targets and collects all embeddings that a slightly similar to at least one of your references/targets (similarity > 0.5). Those embeddings are then used to estimate a umap. In some cases, you start seeing new structure in the umap parts of the umap correspond to irrelevant embeddings (e.g. Membranes). By finding the cluster in the umap that actually corresponds to your target protein can improve the picking!

How to use it
-------------

I assume you have run the reference workflow in this example. But it can be easily used with cluster target embeddings as well.

1. Filter the tomogram embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first select those embeddings that are reasonable close (`-t 0.5`) to our reference embeddings

 .. prompt:: bash $

    tomotwin_tools.py filter_embedding -i embed/tomo_embeddings.temb -m map/map.tmap -t 0.5 -o filter/ --lower --concat

2. Estimate umap
^^^^^^^^^^^^^^^^

 .. prompt:: bash $

    tomotwin_tools.py umap -i filter/tomo_embeddings_filtered_allrefs.temb -o umap/


3. Start napari and select regions of interest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After you have started napari, load the Clustering plugin: :guilabel:`Plugins` -> :guilabel:`napari-tomotwin` -> :guilabel:`Cluster umap embeddings`

Within the plugin select the :file:`.tumap` file in :file:`umap/` folder and press "load".

Select your targets in the umap. By pressing `shift` you can select multiple targets. Save the targets when you are done. I assume you saved it under `cluster_targets/`

4. Map the cluster targets with the tomogram embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 .. prompt:: bash $

    tomotwin_map.py distance -r cluster_targets/cluster_targets.temb -v embed/tomo_embeddings.temb -o map_cluster/


5. Locate the particles
^^^^^^^^^^^^^^^^^^^^^^^

 .. prompt:: bash $

    tomotwin_locate.py findmax -m map_cluster/map.tmap -o locate_refined/


Check your results with the napari-boxmanager :-)
