**This the development version of the code. A streamlined version including instructions how to use it will follow soon.**

# TomoTwin

Particle picking in Tomograms using triplet networks and metric learning

![TomoTwin Logo](resources/images/TomoTwin_black_transparent_cropped.png#gh-light-mode-only)
![TomoTwin Logo](resources/images/TomoTwin_white_transparent_cropped.png#gh-dark-mode-only)


## Installation

To create the necessary conda environment run:

### Default
```
conda create -n tomotwin -c pytorch -c rapidsai -c nvidia python=3.9 pytorch==1.11 torchvision pandas scipy numpy matplotlib pytables rapids=22.04 -c conda-forge
```
### With Optuna support:
```
conda create -n tomotwin -c pytorch -c rapidsai -c nvidia python=3.9 pytorch==1.11 torchvision pandas scipy numpy matplotlib pytables rapids=22.04 optuna mysql-connector-python -c conda-forge
```
For optuna you also need config which should look like the `optuna_config.json` in the resources directory



