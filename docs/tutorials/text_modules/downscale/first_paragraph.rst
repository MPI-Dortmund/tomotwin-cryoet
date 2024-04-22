TomoTwin has been trained on tomograms with a pixel size of 10Å. While in practice we've used it with pixel sizes ranging from 9.2Å to 25.0Å, it's probably often ideal to run it with a pixel size close to 10Å. However, for proteins equal to or larger than the ribosome, we have found that a larger pixel size (e.g. 15Å) works better.  For this you may need to rescale your tomogram. You can do this by Fourier shrinking your tomogram with EMAN2. Suppose you have a tomogram with a pixel size of 5.9359Å. The Fourier shrink factor is then 10Å/5.9359Å = 1.684



.. prompt:: bash $

    e2proc3d.py --apix=5.9359 --fouriershrink=1.684 your_tomo.mrc your_tomo_a10.mrc