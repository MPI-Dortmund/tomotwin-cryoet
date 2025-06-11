.. _dev-info:
Developer information
=====================

This section contains some information snippets for developers. Will be extended in the future.

Reading output formats
**********************

While TomoTwin writes files with various extensions (``.temb``, ``.tmap``, ``.tloc``, ``.tumap``), they are basically all pickled pandas dataframes.
They can all be read by:

.. code:: python

    import pandas as pd
    df = pd.read_pickle("path/to/a/tomotwin/output/file")

In case you modify it, please also check  the `df.attrs` dictionary (and copy it if necessary) of the dataframe. It contains important meta information that is used by TomoTwin.


Implementing new architectures
******************************

Adding new CNN architectures is straightforward in TomoTwin.

1. Add a class for your network in ``modules/networks/`` and implement the interface defined by ``modules/networks/torchmodel.py``
2. Add your new network to the ``network_identifier_map`` dictionary in the ``modules/networks/networkmanager.py`` module.
3. Create a network configuration file like in ``resources/configs/config_siamese.json``. The ``network_config`` entry should match the ``__init__`` method of your new network.

Now you are in principle set to train your network (see ``How to train TomoTwin``).

How to train TomoTwin
*********************

Here we describe how to train the SiameseNet (bad name, as it is actually a tripletnetwork). Hardwarewise, 12GB of GPU memory should be enough.

1. Download training and validation data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training and validation set can be found here:

https://zenodo.org/record/6637456

Download and untar training and validation data.

2. Download siamese network config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You find the configuration file here:

https://github.com/MPI-Dortmund/tomotwin-cryoet/blob/main/resources/configs/config_siamese.json

3. Start the training
^^^^^^^^^^^^^^^^^^^^^

To run it on one GPU for 300 epochs do:

 .. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0 tomotwin_train.py -v path/train/volumes/ --validvolumes path/valid/volumes/ -o out_train -nc path/to/siamese_network.json --epochs 300


How to evaluate TomoTwin
************************

To evaluate TomoTwin you'll need a tomogram to pick on, and ground truth coordinates of particles to compare TomoTwin's pick against. You can find an example tomogram with references and ground truth positions here:

https://zenodo.org/records/15631632

Which you can download with:

 .. prompt:: bash $

    mkdir eval
    cd eval
    wget https://zenodo.org/api/records/15631632/files-archive -O eval.zip
    unzip eval.zip
    mkdir refs
    unzip references.zip
    mv gen01_t*.mrc refs/

Note: this tomogram contains proteins that were not in the training/validation data used to train TomoTwin. Therefore, it is useful to assess the generalization of any models trained using our publicly available training data.

You can also download a json file containing predetermined boxsizes for all of the proteins in the training/validation/evaluation data with:

 .. prompt:: bash $

    wget https://github.com/MPI-Dortmund/tomotwin-cryoet/blob/main/resources/boxsizes.json

To run the evaluation, you should use the reference-based workflow to generate a locate file (replacing the path to your model):

 .. prompt:: bash $

    CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py tomogram -m /path/to/model.pth -v tiltseries_rec.mrc -o ./ -b 256; CUDA_VISIBLE_DEVICES=0,1 tomotwin_embed.py subvolumes -m /path/to/model.pth -v refs/ -b 8 -o ./; tomotwin_map.py distance -r embeddings.temb -v tiltseries_rec_embeddings.temb --refine -o ./; tomotwin_locate.py findmax -m map.tmap -o ./ --write_heatmaps

Then to run the evaluation use:

 .. prompt:: bash $

    tomotwin_scripts_evaluate.py positions -p particle_positions.txt -l located.tloc -s boxsizes.json --optim --stepsize_optim_similarity 0.01

The script will report the picking statistics for each protein in the locate file. The --optim flag will enable metric and size threshold optimization for each protein and the --stepsize_optim_similarity controls the step size for the metric threshold optimisation (default 0.05). Increasing the step size will result in the script running faster, but at the cost of reduced picking optimisation.


