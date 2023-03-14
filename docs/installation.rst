Installation and Download
=========================

Installation
^^^^^^^^^^^^^

We recommend to use the conda client `mamba <https://mamba.readthedocs.io/>`_ for installation. You should use `mambaforge <https://mamba.readthedocs.io/en/latest/installation.html>`_ for installation of mamba. Alternatively you can replace ``mamba`` with ``conda`` in the commands below.

There are three main steps to install TomoTwin:

1. Install TomoTwin
""""""""""""""""""""

.. prompt:: bash $

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

Additionally you need the `napari-tomotwin` plugin [pip]:

.. prompt:: bash $

    pip install napari-tomotwin

3. Link Napari
"""""""""""""""""""

This is an optional step, but for convenience reasons we link an adapted napari call into the tomotwin environment. With that you don't need to switch environments when working with tomotwin. While this is optional, I assume during the tutorials that you did this step. Here is what you need to do:

.. prompt:: bash $

    conda activate tomotwin
    tomotwin_dir=$(realpath $(dirname $(which tomotwin_embed.py)))
    napari_link_file=${tomotwin_dir}/napari_boxmanager
    conda activate napari-tomotwin
    echo -e "#\!/usr/bin/bash\nexport NAPARI_EXE=$(which napari)\nnapari_exe='$(which napari_boxmanager)'\n\${napari_exe} \"\${@}\""> ${napari_link_file}
    ln -rs $(which napari) ${tomotwin_dir}
    chmod +x ${napari_link_file}
    conda activate tomotwin


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
