Installation and Download
=========================

There are three main steps to install TomoTwin:

1. Install TomoTwin
^^^^^^^^^^^^^^^^

.. prompt:: bash $

    conda install mamba -n base -c conda-forge
    mamba create -n tomotwin -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard  optuna mysql-connector-python
    conda activate tomotwin
    pip install git+https://github.com/MPI-Dortmund/tomotwin-cryoet@dev

2. Install Napari
^^^^^^^^^^^^^^

Here we assume that you don't have napari installed. Until the new release from Napari, we need to build it via github as we require the latest changes. Please do:

.. prompt:: bash $

    mamba create -y -n napari-tomotwin -c conda-forge python=3.10
    conda activate napari-tomotwin
    pip install 'napari[all]'
    pip uninstall napari
    pip install git+https://github.com/napari/napari

You can install `napari-boxmanager` via [pip]:

.. prompt:: bash $

    pip install napari-boxmanager

3. Link Napari
^^^^^^^^^^^

This is an optional step, but for convinience reasons we link an adapted napari call into the tomotwin environment. With that you don't need to switch environments when working with tomotwin. While this is optional, I assume during the tutorials that you did this step. Here is what you need to do:

.. prompt:: bash $

    conda activate tomotwin
    napari_link_file=$(realpath $(dirname $(which tomotwin_embed.py))/napari)
    conda activate napari-tomotwin
    echo -e "#\!/usr/bin/bash\nnapari_exe='$(which napari)'\n\${napari_exe} \${@} -w napari-boxmanager __all__" > ${napari_link_file}
    chmod +x ${napari_link_file}

System requirements
^^^^^^^^^^^^^^^^^^^

So far we run it on Ubuntu 20.04 and the following GPUs:

    - NVIDIA V100
    - NVIDIA RTX 2080
    - NVIDIA A100