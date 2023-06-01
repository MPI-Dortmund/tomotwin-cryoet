Developer information
=====================

This section contains some developer information in the future.


Reading output formats
**********************

While TomoTwin writes files with various extensions (".temb", ".tmap", ".tloc", ".tumap"), they are basically all pickled pandas dataframes.
They can all be read by:

.. code:: python

    import pandas as pd
    df = pd.read_pickle("path/to/a/tomotwin/output/file")

In case you modify it, please also check  the `df.attrs` dictionary (and copy it if necessary) of the dataframe. It contains important meta information that is used by TomoTwin.





