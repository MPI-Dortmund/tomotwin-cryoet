Installation and Download
=========================

Installation
^^^^^^^^^^^^^

We recommend to use the conda client `mamba <https://mamba.readthedocs.io/>`_ for installation. You should use the latest `miniforge <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ for installation of mamba. Alternatively you can replace ``mamba`` with ``conda`` in the commands below, but the installation will be much slower.

There are three main steps to install TomoTwin:

1. Install TomoTwin
""""""""""""""""""""

In case you have on old TomoTwin version installed, please remove the old one first with:

.. prompt:: bash $

    mamba env remove -n tomotwin

Next you can create the TomoTwin environment:

.. prompt:: bash $

    mamba env create -n tomotwin -f https://raw.githubusercontent.com/MPI-Dortmund/tomotwin-cryoet/main/conda_env_tomotwin.yml
    conda activate tomotwin
    pip install tomotwin-cryoet

2. Install Napari
"""""""""""""""""""

To install Napari along with the tools and plugins needed for TomoTwin, please do:

.. prompt:: bash $

    mamba env create -n napari-tomotwin -f https://raw.githubusercontent.com/MPI-Dortmund/napari-tomotwin/main/conda_env.yml
    conda activate napari-tomotwin
    pip install napari-tomotwin

If you do not have a computer with an NVIDIA GPU (Mac computers for example), you can install napari without GPU support with:

.. prompt:: bash $

    mamba env create -n napari-tomotwin -f https://raw.githubusercontent.com/MPI-Dortmund/napari-tomotwin/main/conda_env_noGPU.yml
    conda activate napari-tomotwin
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

Update TomoTwin & Napari
^^^^^^^^^^^^^^^^^^^^^^^^
To update an existing TomoTwin installation just do:

.. prompt:: bash $

    mamba env update -n tomotwin -f https://raw.githubusercontent.com/MPI-Dortmund/tomotwin-cryoet/main/conda_env_tomotwin.yml --prune
    conda activate tomotwin
    pip install tomotwin-cryoet
    mamba env update -n napari-tomotwin -f https://raw.githubusercontent.com/MPI-Dortmund/napari-tomotwin/main/conda_env.yml --prune

Download latest model
^^^^^^^^^^^^^^^^^^^^^

:Last update: 09.2023

:Number of proteins: 120

:Link: `Zenodo <https://doi.org/10.5281/zenodo.8137931>`_

System requirements
^^^^^^^^^^^^^^^^^^^

So far we run it on Ubuntu 20.04 and the following GPUs:

    - NVIDIA V100
    - NVIDIA RTX 2080
    - NVIDIA A100
