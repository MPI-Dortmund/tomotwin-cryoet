.. _tutorial-reference:

Tutorial 1: Reference based particle picking
============================================

In this tutorial we describe how to use TomoTwin for picking in tomograms using references.

1. Downscale your Tomogram to 10 Å
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TomoTwin was trained on tomograms with a pixelsize of 10Å. While in practice we've used it with pixel sizes ranging from 9.2 to 13.6, it is probably ideal to run it at a pixel size close to 10Å.  For that you may need to downscale your tomogram. You can do that by fourier shrink your tomogram with EMAN2. Lets say you have a Tomogram with a pixelsize of 5.9359 angstrom. The fouriershrink factor is then 10/5.9359 = 1.684



.. prompt:: bash $

    e2proc3d.py --apix=5.9359 --fouriershrink=1.684 your_tomo.mrc your_tomo_a10.mrc



TomoTwin should be used to pick on tomograms without denoising or lowpass filtering. But you may use these tomograms to find the coordinates of your particle of interest for use as a reference. In this case, you should make sure the denoised/lowpass filtered tomogram has the same pixel size as the one you will pick on.

 .. note::

    **What if my protein is too big for a box size of 37x37x37 pixels?**

    Because TomoTwin was trained on many proteins at once, we needed to find a box size that worked for all proteins. Therefore, all proteins were used with a pixel size of 10Å and a box size of 37 pixels. Because of this, you must extract your reference with a box size of 37 pixels. If your protein is too large for ths box at 10Å/pix (much larger than a ribosome) then you should scale the pixel size of your tomogram until it fits rather than changing the box size. Likewise if your protein is so small that at 10Å/pix it only fills one to two pixels of the box, you should scale your tomogram pixel size until the particle is bigger, however we've found that for proteins down to 100 kDa, 10Å/pix is sufficient for the 37 box.


2. Pick and extract your reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the reference based approach you need, of course, references. To pick them follow the next steps:

1. Open your tomogram in napari

 .. note::

    For easy identification of your reference particle we recommend to use low-pass filter to 60Å and/or denoising (be sure it has the same pixel size of the tomogram you will pick on).

 .. prompt:: bash $

    napari lp60/d01t15.mrc


2. Select :guilabel:`add_layer` tab of the boxmanager toolkit (lower right corner). Press the button for :guilabel:`Create particle layer`.

3. Switch to the :guilabel:`boxmanger` tab and set the :guilabel:`boxsize` to 37, as this gonna be the box size we will use for extraction later on.

4. Press on the :guilabel:`⊕` symbol (upper left corner) to start the picking.

5. Identify a potential reference, choose the slice so that its centered and pick it by clicking in the center of the particle. Continue doing that until you think you have enough references

 .. note::

    **Use multiple references per particle class**

    We recommend to pick multiple (2-3) references per protein of interest, as not all subvolumes work equally well.

    Each reference can be later evaluated separately using the boxmanager, allowing you to decide which gives the best result for each protein of interest

6. Optional: If you want to pick another protein class, we recommend to create a separate particle layer for it (step 2).

7. To save the reference of the selected particle layer (see layer list in napari), click on :guilabel:`File` -> :guilabel:`Save Selected Layer(s)`. Create a new folder by right click in the dialog and name it for example 'coords'. Now select as :guilabel:`Files of type` the entry :guilabel:`Box Manager`. Use the filename `reference.coords` and press :guilabel:`Save`.

8. Finally, use the ``tomotwin_tools.py extractref`` script to extract a subvolume from the tomogram (the original, not the denoised / low pass filtered) at the coordinates for each reference. If there are multiple references you would like to pick in the tomogram, repeat this process multiple times giving a new output folder each time.

 .. prompt:: bash $

    tomotwin_tools.py extractref --tomo tomo/your_tomo_a10.mrc --coords path/to/references.coords --out reference/ --filename protein_a

You will find your extracted references in `reference/protein_a_X.mrc` where X is a running number.


3. Embed your Tomogram
^^^^^^^^^^^^^^^^^^^^^^

I assume that you already have downloaded the general model.

To embed your tomogram using two GPUs do:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py tomogram -m LATEST_TOMOTWIN_MODEL.pth -v your_tomo_a10.mrc -b 256 -o your_tomo_a10/embed/tomo/ -w 37 -s 2


4. Embed your reference
^^^^^^^^^^^^^^^^^^^^^^^

Now you can embed your reference:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py subvolumes -m LATEST_TOMOTWIN_MODEL.pth -v reference/*.mrc -b 12 -o your_tomo_a10/embed/reference/


5. Map your tomogram
^^^^^^^^^^^^^^^^^^^^

The map command will calculate the pairwise distances/similarity between the references and the subvolumes and generate a localization map:

.. prompt:: bash $

    tomotwin_map.py distance -r your_tomo_a10/embed/reference/embeddings.temb -v your_tomo_a10/embed/tomo/your_tomo_a10_embeddings.temb -o your_tomo_a10/classify/

6. Localize potential particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run `tomotwin_locate` to locate particles:

.. prompt:: bash $

    tomotwin_locate.py findmax -p your_tomo_a10/classify/map.tmap -o your_tomo_a10/locate/

.. note::

    **Similarity maps**

    In the output folder :file:`out/locate/` you will find a similarity map.mrc for each reference - just in case you are interested, this is akin to a location confidence heatmap for each protein.

7. Inspect your particles with the boxmanager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate the your napari environment to inspect your selected particles. I assume the environment is called `napari`.

.. prompt:: bash $

    conda activate napari

Open your particles with the following command or drag the files into an open napari window:

.. prompt:: bash $

    napari tomo/your_tomo_a10.mrc out/locate/located.tloc -w napari-boxmanager

.. image:: ../img/tutorial_1/start.png
   :width: 650

The example shown here is from the SHREC competition. In the table on the right you see 12 references. I selected the :guilabel:`model_8_5MRC_86.mrc`, which is a ribosome.
Below the table, you need to adjust the :guilabel:`metric min` and :guilabel:`size min` thresholds until you like the results. After the optimization is done the result might look similar to this:

.. image:: ../img/tutorial_1/after_optim.png
   :width: 650

In the left panel, select the references you would like to pick (ctrl click on windows, cmd click on mac to select multiple). You can now press :guilabel:`File` -> :guilabel:`Save selected Layer(s)`. In the dialog, change the :guilabel:`Files of type` to  :guilabel:`Box Manager`. Choose filename like :guilabel:`selected_coords.tloc`. Make sure that the file ending is :file:`.tloc`.

To convert the :file:`.tloc` file into :file:`.coords` you need to run

.. prompt:: bash $

    tomotwin_pick.py -l coords.tloc -o coords/

You will find coordinate file for each reference in :file:`.coords` format in the :file:`coords/` folder.

8. Scale your coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

After step 7 you have the coordinates for each protein of interest in your tomogram. Assuming you downscaled your tomogram in step 1, you now need to scale your coordinates to the pixel size you would like to use for extraction. Assuming that you would like to extract from tomograms with a pixel size of 5.936 A/pix, then the command would be:

.. prompt:: bash $

    tomotwin_tools.py scale_coordinates --coords coords/your_coords_file.coords --tomotwin_pixel_size 10 --extraction_pixel_size 5.9356 --out multi_refs_0_a5936.coords


