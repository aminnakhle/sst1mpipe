# SST-1M pipeline

The [sst1mpipe](https://github.com/SST-1M-collaboration/sst1mpipe) repository serves for the analysis of <b>SST-1M prototype</b>. It is based on [ctapipe](https://github.com/cta-observatory/ctapipe).

## Installation

The advanced package and environment management system, [Anaconda](https://www.anaconda.com/distribution/#download-section), [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://anaconda.org/conda-forge/mamba), is needed to be installed first.

The Mamba is recomended due to some (quite often occured) stucks at solving environment on Anaconda. Up to now Mamba works well.

Set up environment based on Mamba
(also described in https://github.com/conda-forge/miniforge#mambaforge)

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```
or
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```



### For users
- download stable version of <b>sst1mpipe</b> (latest version = 0.0.0)
- create and activate <b>conda</b> environment
- install <b>sst1mpipe</b>
```
SST1MPIPE_VER=0.0.0

wget https://raw.githubusercontent.com/SST-1M-collaboration/sst1mpipe/v$SST1MPIPE_VER/environment.yml

conda env create -n sst1m -f environment.yml
conda activate sst1m
pip install sst1mpipe==$SST1MPIPE_VER

rm environment.yml
```

### For developers
- download latest development vertsion from git repository
- create and activate <b>conda</b> environment
- install <b>sst1mpipe</b>
```
git clone git@github.com:SST-1M-collaboration/sst1mpipe.git
conda env create -f sst1mpipe/environment.yml
conda activate sst1m-dev
pip install -e sst1mpipe
```

