from setuptools import setup
import datetime
import sys
import codecs
import os
import re

pthversion = os.path.join(os.path.dirname(__file__), "VERSION.txt")
if sys.argv[1] == "sdist":
    __version__ = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    f = open(pthversion, 'w')
    f.write(__version__)
    f.close()
else:
    f = open(pthversion, 'r')
    __version__ = f.read()

def find_version(*file_paths):
    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, *parts), 'r') as fp:
            return fp.read()

    print("==========================================",sys.argv)
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='tomotwin',
    version=find_version("tomotwin", "__init__.py")+"post"+__version__,
    python_requires='>=3.7.0',
    packages=[
        'scripts','tomotwin','tomotwin.modules',
        'tomotwin.modules.networks',
        'tomotwin.modules.training',
        'tomotwin.modules.inference',
        'tomotwin.modules.tools',
        'tomotwin.modules.common',
        'tomotwin.modules.common.findmax',
    ],
    url='',
    license='MIT',
    author='Gavin Rice, Thorsten Wagner',
    install_requires=[
        "mrcfile",
        "tensorboard",
        "numpy >= 1.20.0",
        "scikit-learn",
        "scikit-image",
        "pystardb==0.3.1",
        "pandas >= 1.3",
        "pytorch-metric-learning",
        "cupy-cuda116",
        "numba",
        "tabulate"
    ],
    author_email='thorsten.wagner@mpi-dortmund.mpg.de',
    description='Picking procedure for cryo em tomography',
    long_description='',
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'tomotwin_train.py = tomotwin.train_main:_main_',
            'tomotwin_optuna.py = tomotwin.train_optuna:_main_',
            'tomotwin_embed.py = tomotwin.embed_main:_main_',
            'tomotwin_map.py = tomotwin.map_main:_main_',
            'tomotwin_locate.py = tomotwin.locate_main:_main_',
            'tomotwin_pick.py = tomotwin.pick_main:_main_',
            'tomotwin_scripts_evaluate.py = scripts.evaluation:_main_',
            'tomotwin_scripts_lasso.py = scripts.lasso:_main_',
            'tomotwin_dev_matrix.py = scripts.pdb_similarity_matrix:_main_',
            'tomotwin_dev_json2tloc.py = scripts.json2tloc:_main_',
            'tomotwin_dev_molmapbbox.py = scripts.molmapsbbox:_main_',
            'tomotwin_tools.py = tomotwin.tools_main:_main_'
        ]},
)
