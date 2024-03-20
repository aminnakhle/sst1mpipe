.. _introduction:

Introduction
============

Installation
------------

The advanced package and environment management system, `Anaconda <https://www.anaconda.com/distribution/#download-section>`_, `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Mamba <https://anaconda.org/conda-forge/mamba>`_, is needed to be installed first.

The Mamba is recomended due to some (quite often occured) stucks at solving environment on Anaconda. Up to now Mamba works well.

Set up environment based on Mamba 

.. note::

    For more details see https://github.com/conda-forge/miniforge#mambaforge

.. code-block:: console

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh

or

.. code-block:: console

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh

For users
~~~~~~~~~

- download stable version of **sst1mpipe** (latest version = 0.4.0)
- create and activate **conda** environment
- install **sst1mpipe**

.. code-block:: console

    SST1MPIPE_VER=0.4.0

    wget https://raw.githubusercontent.com/SST-1M-collaboration/sst1mpipe/v$SST1MPIPE_VER/environment.yml

    conda env create -n sst1m -f environment.yml
    conda activate sst1m
    pip install sst1mpipe==$SST1MPIPE_VER

    rm environment.yml


For developers
~~~~~~~~~~~~~~

- download latest development version from git repository
- create and activate **conda** environment
- install **sst1mpipe**

.. code-block:: console

    git clone git@github.com:SST-1M-collaboration/sst1mpipe.git
    conda env create -f sst1mpipe/environment.yml
    conda activate sst1m-dev
    pip install -e sst1mpipe


Setup pre-installed conda environment on **Calculus**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If one prefers to work on **Calculus**, he/she may skip the pipeline installation completely and 
only activate preinstalled environment: 

.. code-block:: console

    source /data/work/analysis/software/mambaforge/etc/profile.d/conda.sh
    conda activate /data/work/analysis/software/mambaforge/envs/sst1m-$SST1MPIPE_VER


Analysis basics
---------------

TBD