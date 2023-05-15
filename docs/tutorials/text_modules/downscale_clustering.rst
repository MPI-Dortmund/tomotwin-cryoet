TomoTwin was trained on tomograms with a pixelsize of 10Å. While in practice we've used it with pixel sizes ranging from 9.2Å to 25.0Å, it is probably ideal to run it at a pixel size close to 10Å.  For that you may need to downscale your tomogram. You can do that by fourier shrink your tomogram with EMAN2. Lets say you have a Tomogram with a pixelsize of 5.9359Å. The fouriershrink factor is then 10Å/5.9359Å = 1.684



.. prompt:: bash $

    e2proc3d.py --apix=5.9359 --fouriershrink=1.684 your_tomo.mrc your_tomo_a10.mrc



TomoTwin should be used to pick on tomograms without denoising or lowpass filtering. But you may use these tomograms for visualizing the picks in Napari. In this case, you should make sure the denoised/lowpass filtered tomogram has the same pixel size as the one you will pick on (downscaling it if necessary).

.. admonition:: **What if my protein is too big for a box size of 37x37x37 pixels?**
    
    Because TomoTwin was trained on many proteins at once, we needed to find a box size that worked for all proteins. Therefore, all proteins were used with a pixel size of 10Å and a box size of 37 pixels. Because of this, you must extract your reference with a box size of 37 pixels. If your protein is too large for ths box at 10Å/pix (much larger than a ribosome) then you should scale the pixel size of your tomogram until it fits rather than changing the box size. Likewise if your protein is so small that at 10Å/pix it only fills one to two pixels of the box, you should scale your tomogram pixel size until the particle is bigger, however we've found that for proteins down to 100 kDa, 10Å/pix is sufficient for the 37 box.
