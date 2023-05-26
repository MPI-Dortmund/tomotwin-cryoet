TomoTwin was trained on tomograms with a pixelsize of 10Å. While in practice we've used it with pixel sizes ranging from 9.2Å to 25.0Å, it is probably ideal to run it at a pixel size close to 10Å.  For that you may need to downscale your tomogram. You can do that by fourier shrink your tomogram with EMAN2. Lets say you have a Tomogram with a pixelsize of 5.9359Å. The fouriershrink factor is then 10Å/5.9359Å = 1.684



.. prompt:: bash $

    e2proc3d.py --apix=5.9359 --fouriershrink=1.684 your_tomo.mrc your_tomo_a10.mrc