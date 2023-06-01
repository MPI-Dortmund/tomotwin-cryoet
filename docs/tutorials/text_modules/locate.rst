To locate potential particles positions for each target run:

.. prompt:: bash $

    tomotwin_locate.py findmax -m out/map/map.tmap -o out/locate/

.. hint:: **Similarity maps**

    You can add the option ``--write_heatmaps`` to the locate command. If you do this you will find a similarity map for each reference in :file:`your_tomo_a10/locate/` - just in case you are interested, this is akin to a location confidence heatmap for each protein.
