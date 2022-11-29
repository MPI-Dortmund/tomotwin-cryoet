Installation and Download
=========================

Installation
^^^^^^^^^^^^^

There are three main steps to install TomoTwin:

1. Install TomoTwin
""""""""""""""""""""

.. prompt:: bash $

    conda install mamba -n base -c conda-forge
    mamba create -n tomotwin -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard  optuna mysql-connector-python
    conda activate tomotwin
    pip install tomotwin-cryoet

2. Install Napari
"""""""""""""""""""

Here we assume that you don't have napari installed. Please do:

.. prompt:: bash $
    mamba create -y -n napari-tomotwin -c conda-forge python=3.10 napari=0.4.17 pyqt pip
    conda activate napari-tomotwin

You can install `napari-boxmanager` via [pip]:

.. prompt:: bash $

    pip install napari-boxmanager

3. Link Napari
"""""""""""""""""""

This is an optional step, but for convinience reasons we link an adapted napari call into the tomotwin environment. With that you don't need to switch environments when working with tomotwin. While this is optional, I assume during the tutorials that you did this step. Here is what you need to do:

.. prompt:: bash $

    conda activate tomotwin
    napari_link_file=$(realpath $(dirname $(which tomotwin_embed.py))/napari)
    conda activate napari-tomotwin
    echo -e "#\!/usr/bin/bash\nnapari_exe='$(which napari)'\n\${napari_exe} \"\${@}\" -w napari-boxmanager __all__" > ${napari_link_file}
    chmod +x ${napari_link_file}

Download latest model
^^^^^^^^^^^^^^^^^^^^^

:Last update: 05.2022

:Number of proteins: 120

:Link: `https <https://ftp.gwdg.de/pub/misc/sphire/TomoTwin/models/tomotwin_model_p120_052022_loss.pth>`_

System requirements
^^^^^^^^^^^^^^^^^^^

So far we run it on Ubuntu 20.04 and the following GPUs:

    - NVIDIA V100
    - NVIDIA RTX 2080
    - NVIDIA A100