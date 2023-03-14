After step 7 you have the coordinates for each protein of interest in your tomogram. Assuming you downscaled your tomogram in step 1, you now need to scale your coordinates to the pixel size you would like to use for extraction. Assuming that you would like to extract from tomograms with a pixel size of 5.936 A/pix, then the command would be:

.. prompt:: bash $

    tomotwin_tools.py scale_coordinates --coords coords/your_coords_file.coords --tomotwin_pixel_size 10 --extraction_pixel_size 5.9356 --out multi_refs_0_a5936.coords


