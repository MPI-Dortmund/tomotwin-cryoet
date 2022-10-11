**This the development version of the code. A streamlined version including instructions how to use it will follow soon.**

**Documentation: https://tomotwin-cryoet.readthedocs.io/en/latest/index.html#**



# TomoTwin

Particle picking in Tomograms using triplet networks and metric learning

![TomoTwin Logo](resources/images/TomoTwin_black_transparent_cropped.png#gh-light-mode-only)
![TomoTwin Logo](resources/images/TomoTwin_white_transparent_cropped.png#gh-dark-mode-only)


## Installation

**!!!** The installation of the napari-boxmanager requires at least napari version `0.4.16.rc8`! **!!!**
If the version is not yet available use:

    conda create -y -n napari-env -c conda-forge python=3.9
    conda activate napari-env
    pip install 'napari[all]'
    pip uninstall napari
    pip install git+https://github.com/napari/napari

**!!!** The following is not yet working, pypi upload follows soon **!!!**

You can install `napari-boxmanager` via [pip]:

    pip install napari-boxmanager

### Default
```
conda install mamba -n base -c conda-forge
mamba create -n tomotwin_t12 -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard
pip install .
conda remove --force cupy
```
### With Optuna support:
```
conda install mamba -n base -c conda-forge
mamba create -n tomotwin_t12 -c pytorch -c rapidsai -c nvidia -c conda-forge python=3.9 pytorch==1.12 torchvision pandas scipy numpy matplotlib pytables cuML=22.06 cudatoolkit=11.6 'protobuf>3.20' tensorboard optuna mysql-connector-python
pip install .
conda remove --force cupy
```
For optuna you also need config which should look like the `optuna_config.json` in the resources directory



