Strategy 3: Manually refining UMAPs from selected embeddings (Napari GPU workaround)
========================================================

When to use it
--------------

Use this strategy as a workaround to refine UMAP embeddings if you don't use Napari on a machine with GPUs.

What it does
------------

This strategy allows you to manually recalculate UMAP embeddings that you can then reload in the Napari GUI.

How to use it
-------------

1. Run the clustering workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This strategy works with the clustering workflow. Therefore run the clustering workflow including cluster selection in Napari, save the clusters that you would like to recalculate.

2. Recalculate the UMAP from the embeddings from each cluster of interest.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we manually recalculate the UMAP for each cluster we would like to refine. For example if you labeled your cluster cool_protein:

 .. prompt:: bash $

    tomotwin_tools.py umap -i out/clustering/embeddings_cool_protein.temb -o out/clustering/refined/


3. Visualize the refined UMAP in Napari
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you can load the refined UMAP in napari-boxmanager and look for subclusters that pick your protein of interest better


