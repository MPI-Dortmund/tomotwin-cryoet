Installation and Download
=========================

Here are some installation instructions.


System requirements
^^^^^^^^^^^^^^^^^^^

Where was it tested on


Install TomoTwin
^^^^^^^^^^^^^^^^

.. prompt:: bash $

    conda install mamba -n base -c conda-forge
    mamba create -n tomotwin -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard  optuna mysql-connector-python
    pip install .
    conda remove --force cupy

Install Napari
^^^^^^^^^^^^^^

Here we assume that you don't have napari installed. Until the new release from Napari, we need to build it via github as we require the latest changes. Please do:

.. prompt:: bash $
    conda create -y -n napari-env -c conda-forge python=3.9
    conda activate napari-env
    pip install 'napari[all]'
    pip uninstall napari
    pip install git+https://github.com/napari/napari

You can install `napari-boxmanager` via [pip]:

.. prompt:: bash $
    pip install napari-boxmanager

Link Napari
^^^^^^^^^^^

This is an optional step, but for convinience reasons we link an adapted napari call into the tomotwin environment. With that you don't need to switch environments when working with tomotwin. While this is optional, I assume during the tutorials that you did this step. Here is what you need to do:

.. prompt:: bash $

    conda activate tomotwin
    napari_link_file=$(realpath $(dirname $(which tomotwin_embed.py))/napari)
    conda activate napari-tomotwin
    echo -e "#\!/usr/bin/bash\nnapari_exe='$(which napari)'\n\${napari_exe} \${@} -w napari-boxmanager __all__" > ${napari_link_file}
    chmod +x ${napari_link_file}

