from setuptools import setup
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tomotwin-cryoet',
    use_scm_version={'write_to': 'tomotwin/__init__.py'},
    setup_requires=['setuptools_scm'],
    python_requires='>=3.7.0',
    packages=[
        'scripts','tomotwin','tomotwin.modules',
        'tomotwin.modules.networks',
        'tomotwin.modules.training',
        'tomotwin.modules.inference',
        'tomotwin.modules.tools',
        'tomotwin.modules.common',
        'tomotwin.modules.common.findmax',
        'tomotwin.modules.common.io',
    ],
    url='https://github.com/MPI-Dortmund/tomotwin-cryoet',
    license='Mozilla Public License Version 2.0',
    author='Gavin Rice, Thorsten Wagner, Markus Stabrin',
    install_requires=[
        "mrcfile",
        "tensorboard",
        "numpy >= 1.20.0",
        "scikit-learn",
        "scikit-image",
        "pystardb==0.3.1",
        "pandas >= 1.3",
        "pytorch-metric-learning",
        "numba",
        "tabulate",
        "setuptools_scm",
        "tqdm"
    ],
    author_email='thorsten.wagner@mpi-dortmund.mpg.de',
    description='Picking procedure for cryo em tomography',
    long_description=long_description,
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
