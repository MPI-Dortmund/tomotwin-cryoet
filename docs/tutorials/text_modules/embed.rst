I assume that you already have downloaded the general model.

To embed your tomogram using two GPUs and batchsize of 256 use:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py tomogram -m LATEST_TOMOTWIN_MODEL.pth -v your_tomo_a10.mrc -b 256 -o out/embed/tomo/ -s 2

.. hint:: **The batchsize parameter**

    To have your tomograms embedded as quick as possible, you should choose a batchsize that utilize your GPU memory as much as possible. However, if you choose it too big, you might run into memory problems. In those cases play around with different batch sizes and check the memory usage with `nvidia-smi`.

.. hint:: **Strategy: Speed up embedding calculation using a mask**

    Using masks can dramatically speed up the embedding calculation. It can also improve the estimated umaps!

    Check out the :ref:`corresponding strategy <strategy-02>`!