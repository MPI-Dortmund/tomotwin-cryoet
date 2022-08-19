**This the development version of the code. A streamlined version including instructions how to use it will follow soon.**

# TomoTwin

Particle picking in Tomograms using triplet networks and metric learning

![TomoTwin Logo](resources/images/TomoTwin_black_transparent_cropped.png#gh-light-mode-only)
![TomoTwin Logo](resources/images/TomoTwin_white_transparent_cropped.png#gh-dark-mode-only)


## Installation

To create the necessary conda environment run:

### Default
```
conda install -c conda-forge mamba
mamba create -n tomotwin_t12 -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard
pip install .
conda remove --force cupy
```
### With Optuna support:
```
conda install -c conda-forge mamba
mamba create -n tomotwin_t12 -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard optuna mysql-connector-python
pip install .
conda remove --force cupy
```
For optuna you also need config which should look like the `optuna_config.json` in the resources directory



