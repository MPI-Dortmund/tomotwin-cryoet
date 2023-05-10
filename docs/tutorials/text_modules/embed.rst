I assume that you already have downloaded the general model.

To embed your tomogram using two GPUs do:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py tomogram -m LATEST_TOMOTWIN_MODEL.pth -v your_tomo_a10.mrc -b 256 -o out/embed/tomo/ -s 2