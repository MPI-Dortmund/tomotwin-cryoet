Tutorial 1: Reference based particle picking
============================================

In this tutorial we describe how to use TomoTwin for picking in tomograms using references.

1. Downscale your Tomogram to 10 Å
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can do that by fourier shrink your tomogram with EMAN2. Lets say you have a Tomogram with a pixelsize of 5.9359 angstrom. The fouriershrink factor is then 10/5.9359 = 1.684

.. prompt:: bash $

    e2proc3d.py --apix=5.9359 --fouriershrink=1.684 your_tomo.mrc your_tomo_a10.mrc

Don't denoise your tomogram that you want to use. TomoTwin was trained on unfiltered data and therefore should be applied to unfiltered data as well.


2. Pick and extract your reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. To use a subvolume within the tomogram as a reference, open the tomogram in imod:

 .. prompt:: bash $

    imod tomo/your_tomo_a10.mrc

 For easy identification we recommend to use low-pass filter to 60 angstroms and/or denoising.


.. admonition::

    **Use multiple references per particle class**

    We recommend to pick multiple (2-3) references per particle class, as not all subvolumes work equally well.


2. As :guilabel:`Mode` select :guilabel:`model` instead of :guilabel:`movie`, navigate to the central slice of the particle you would like to use as a reference, middle click to place a point on the center of the particle. You can use :guilabel:`Edit` -> :guilabel:`Object` -> :guilabel:`Type` and increase the :guilabel:`Sphere radius for points` to visualize the box that will be used for extraction (radius 18 or 19).


3. Press :kbd:`s` to open the model saving window and save the model as something like "reference_a.mod".


4. Exit imod and use the command

 .. prompt:: bash $

    model2point -inp reference_a.mod -ou reference_a.coords

 command to convert to a coords file.

5. Finally, use the `tomotwin_tools.py extractref` script to extract a box from the tomogram (the original, not the denoised/lp60) at the coordinates for the reference. If there are multiple references you would like to pick in the tomogram, repeat this process multiple times changing the name of the reference each time.

 .. prompt:: bash $

    tomotwin_tools.py extractref --tomo path/to/tomo.mrc --coords path/to/coords.coords --out reference/ --filename reference_a

 You will find your reference in `reference/reference_a.mrc`

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

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py subvolumes -m tomotwin_model_p120_052022.pth -v reference/ref.mrc -b 12 -o out/embed/reference/


5. Map your tomogram
^^^^^^^^^^^^^^^^^^^^

Map will calculate the pairwise distances/similarity between the references and the subvolumes and therefore a localization map:

.. prompt:: bash $

    tomotwin_map.py distance -r out/embed/reference/embeddings.temb -v out/embed/tomo/d01t04_embeddings.temb -o out/classify/tomo_apof/

6. Localize potential particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run `tomotwin_locate` to locate particles:

.. prompt:: bash $

    tomotwin_locate.py findmax -p out/classify/tomo_apof/map.tmap -o out/locate/

7. Inspect your particles with the boxmanager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate the your napari environment to inspect your selected particles. I assume