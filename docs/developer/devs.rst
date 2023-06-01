Developer information
=====================

This section contains some developer information in the future.


Reading output formats
**********************

While TomoTwin write several file endings (".temb", ".tmap", ".tloc", ".tumap"), they are basically all pickled pandas dataframe.
They can all be read by:

```python
df = pd.read_pickle("path/to/a/tomotwin/output/file")
```

