I assume that you already have downloaded the general model.

To embed your tomogram using two GPUs and batchsize of 256 use:

.. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py tomogram -m LATEST_TOMOTWIN_MODEL.pth -v your_tomo_a10.mrc -b 256 -o out/embed/tomo/ -s 2

.. hint:: **The batchsize parameter**

    To have your tomograms embedded as quick as possible, you should choose a batchsize that utilize your GPU memory as much as possible. However, if you choose it too big, you might run into memory problems. In those cases play around with different batch sizes and check the memory usage with `nvidia-smi`.

.. hint:: **Speed up embedding using a mask**

    With TomoTwin 0.5, the emedding command supports the use of masks. With masks you can define which regions of your tomogram get actually embedded and therefore speedup the embbeding.
    We also provide new tools that calculates mask that excludes areas that probably does not contain any protein. You can run it with:

    .. prompt:: bash $

        tomotwin_tools.py embedding_mask -i your_tomo_a10.mrc -o out/mask/

    The mask you find there can be used when running ``tomotwin_embed.py`` using the argument ``--mask``.
    As this is still experimental, please check if the masks do not exclude any important areas. You can do that easiliy with napari by opening the tomogram and your mask and then change the opacity of your mask:

    .. prompt:: bash $

        napari your_tomo_a10.mrc out_mask/your_tomo_a10_mask.mrc