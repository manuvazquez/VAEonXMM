#!/bin/bash

MANAGER="mamba"

NAME=trap

LIBRARIES=(
    python">="3.12
    ipdb # debugging
    nbdime
    jupyterlab">"4
    astropy astroquery
    matplotlib
    plotly ipywidgets python-kaleido
    altair
    scikit-learn
    scipy
    pandas pyarrow
	mkdocstrings-python python-markdown-math
    pytorch torchvision
	jaxlib=*=*cuda* jax cuda-nvcc
    equinox optax">="0.2 chex">="0.1.85
)

CHANNELS=(
    defaults
    conda-forge
    nvidia # required for CUDA-enabled jax
    # pytorch # just needed for cpuonly
)

# ---

COLOR="\033[40m\033[32m"
UNCOLOR="\033[0m"


# ------------ setup

# only required if "anaconda" is not in the path
source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh

# from https://stackoverflow.com/a/9429887/3967334
LIBRARIES_CONCATENATED=$(IFS=" " ; echo "${LIBRARIES[*]}")

# from https://stackoverflow.com/a/17841619/3967334
function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

CHANNELS_CONCATENATED=$(join_by ' -c ' "${CHANNELS[@]}")

# ------------ installation

$MANAGER create --yes -n $NAME $LIBRARIES_CONCATENATED -c $CHANNELS_CONCATENATED

# ------------ pip

conda activate $NAME

# pip stuff here....
pip install mlflow # apparently, conda installation is not working
pip install -e .

# ------------

echo -e new environment is \"$COLOR$NAME$UNCOLOR\"

# # the environment is exported into a yaml file
# conda env export --no-builds --from-history -f environment.yml
