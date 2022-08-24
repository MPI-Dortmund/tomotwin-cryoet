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
