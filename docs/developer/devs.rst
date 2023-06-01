Developer information
=====================

This section contains some information snippets for developers. Will be extended in the future.

Reading output formats
**********************

While TomoTwin writes files with various extensions (".temb", ".tmap", ".tloc", ".tumap"), they are basically all pickled pandas dataframes.
They can all be read by:

.. code:: python

    import pandas as pd
    df = pd.read_pickle("path/to/a/tomotwin/output/file")

In case you modify it, please also check  the `df.attrs` dictionary (and copy it if necessary) of the dataframe. It contains important meta information that is used by TomoTwin.


Implementing new architectures
******************************

Adding new CNN architectures is straightforward in TomoTwin.

1. Add a class for your network in ```modules/networks/`` and implement the interface defined by ```modules/networks/torchmodel.py```
2. Add your newly defined made network to the ```network_identifier_map``` dictonary in the ```modules/networks/networkmanager.py``` module.
3. Create a network configuration file like in ```resources/configs/config_siamese.json```. The ```network_config``` entry should match your ```__init__``` method your new network.

Now you are in principle set to train your network (see ```How to train TomoTwin```).

How to train TomoTwin
*********************

Training and validation set can be found here:

https://zenodo.org/record/6637456

More will follow soon :-)


How to evaluate TomoTwin
************************

Will follow soon :-)

