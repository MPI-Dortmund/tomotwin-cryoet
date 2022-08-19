Tutorial 1: Reference based particle picking
============================================

In this tutorial we describe how to use TomoTwin for picking in tomograms using references.

1. Downscale your Tomogram to 10 Å
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TomoTwin was trained on data with a pixelsize of 10A. While it is also work with data with a lower or higher pixel size it is probably ideal to run it on data with the same pixelsize. For that you may need to downscale your tomogram. You can do that by fourier shrink your tomogram with EMAN2. Lets say you have a Tomogram with a pixelsize of 5.9359 angstrom. The fouriershrink factor is then 10/5.9359 = 1.684

.. prompt:: bash $

    e2proc3d.py --apix=5.9359 --fouriershrink=1.684 your_tomo.mrc your_tomo_a10.mrc

Don't denoise your tomogram that you want to use. TomoTwin was trained on unfiltered data and therefore should be applied to unfiltered data as well.


2. Pick and extract your reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the reference based approch you need, of course, references. To pick them follow the next steps:

1. To use a subvolume within the tomogram as a reference, open the tomogram in imod:

 .. prompt:: bash $

    imod tomo/your_tomo_a10.mrc

 For easy identification we recommend to use low-pass filter to 60 angstroms and/or denoising.


 .. note::

    **Use multiple references per particle class**

    We recommend to pick multiple (2-3) references per particle class, as not all subvolumes work equally well.


2. As :guilabel:`Mode` select :guilabel:`model` instead of :guilabel:`movie`, navigate to the central slice of the particle you would like to use as a reference, middle click to place a point on the center of the particle. You can use :guilabel:`Edit` -> :guilabel:`Object` -> :guilabel:`Type` and increase the :guilabel:`Sphere radius for points` to visualize the box that will be used for extraction (radius 18 or 19).


3. Press :kbd:`s` to open the model saving window and save the model as something like :file:`references.mod`.


4. Exit imod and use the command

 .. prompt:: bash $

    model2point -inp references.mod -ou references.coords

 command to convert to a coords file.

5. Finally, use the `tomotwin_tools.py extractref` script to extract a box from the tomogram (the original, not the denoised/lp60) at the coordinates for the reference. If there are multiple references you would like to pick in the tomogram, repeat this process multiple times changing the name of the reference each time.

 .. prompt:: bash $

    tomotwin_tools.py extractref --tomo tomo/your_tomo_a10.mrc --coords path/to/references.coords --out reference/ --filename references

 You will find your extracted references in `reference/references_X.mrc` where X is a running number.

3. Embed your Tomogram
^^^^^^^^^^^^^^^^^^^^^^

Download the latest tomotwin model here:

https://owncloud.gwdg.de/index.php/s/vfjKoBZc4YtPaGT

To embed your tomogram using two GPUs do:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py tomogram -m tomotwin_model_p120_052022.pth -v your_tomo_a10.mrc -b 256 -o out/embed/ -w 37 -s 2

4. Embed your reference
^^^^^^^^^^^^^^^^^^^^^^^

Now you can embed your reference:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py subvolumes -m tomotwin_model_p120_052022.pth -v reference/*.mrc -b 12 -o out/embed/reference/


5. Map your tomogram
^^^^^^^^^^^^^^^^^^^^

The map command will calculate the pairwise distances/similarity between the references and the subvolumes and therefore a localization map:

.. prompt:: bash $

    tomotwin_map.py distance -r out/embed/reference/embeddings.temb -v out/embed/tomo/your_tomo_a10_embeddings.temb -o out/map/

6. Localize potential particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run `tomotwin_locate` to locate particles:

.. prompt:: bash $

    tomotwin_locate.py findmax -m out/map/map.tmap -o out/locate/

.. note::

    **Similarity maps**

    In the output folder :file:`out/locate/` you will find a similarity map for each reference - just in case you are interested.

7. Inspect your particles with the boxmanager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate the your napari environment to inspect your selected particles. I assume the environment is called `napari`.

.. prompt:: bash $

    conda activate napari

Open your particles with the following command:

.. prompt:: bash $

    napari tomo/your_tomo_a10.mrc out/locate/located.tloc -w napari-boxmanager

.. image:: ../img/tutorial_1/start.png
   :width: 650

The example shown here is from the SHREC competition. In the table on the right you see 12 references. I selected the :guilabel:`model_8_5MRC_86.mrc`, which is a ribosome.
Below the table, you need to adjust the :guilabel:`metric min` and :guilabel:`size min` thresholds until you like the results. After the optimization is done the result might look similar to this:

.. image:: ../img/tutorial_1/after_optim.png
   :width: 650

You can now press :guilabel:`File` -> :guilabel:`Save selected Layer(s)`. In the dialog, change the :guilabel:`Files of type` to  :guilabel:`Box Manager`. Choose filename like :guilabel:`coords.tloc`. Make sure that the file ending is :file:`.tloc`.

To convert the :file:`.tloc` file into :file:`.coords` you need to run

.. prompt:: bash $

    tomotwin_pick.py -l coords.tloc -o dotcoords/

You will find coordinate file in :file:`.coords` format in the :file:`dotcoords/` folder.

8. Scale your coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

After step 7 you have the coordinates for your tomogram. If you downscaled your tomogram in step 1, you need to scale your coordinates. I assume that your initial tomogram had a pixelsize of 5.9359, then the command would be:

.. prompt:: bash $

    tomotwin_tools.py scale_coordinates --coords dotcoords/your_coords_file.coords --tomotwin_pixel_size 10 --extraction_pixel_size 5.9359 --out multi_refs_0_bin2.coords


