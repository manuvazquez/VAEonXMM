#!/bin/bash

MANAGER="mamba"

NAME=trap

LIBRARIES=(
    python=3
    ipdb # debugging
    nbdime
    jupyterlab">"4
    astropy astroquery
    matplotlib
    plotly ipywidgets python-kaleido
    altair
    scikit-learn
    scipy"<"1.13 # scipy.linalg.tril error
    pandas pyarrow
	mkdocstrings-python python-markdown-math
    # pytorch torchvision
    pytorch torchvision cpuonly # apparently, using the above line without a GPU is a problem
    jax equinox optax
    papermill
)

CHANNELS=(
    defaults
    conda-forge
    # nvidia # required for CUDA-enabled jax
    pytorch # just needed for cpuonly
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
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install astroquery
pip install -e .

# ------------

echo -e new environment is \"$COLOR$NAME$UNCOLOR\"

# # the environment is exported into a yaml file
# conda env export --no-builds --from-history -f environment.yml
