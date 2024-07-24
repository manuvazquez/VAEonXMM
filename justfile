home_dir := env_var('HOME')
# tmp_dir := '/tmp/'

light_curves_dir := join(home_dir, 'Sync/research/light curves')
trapallada_dir := join(home_dir, 'Sync/git/trapallada')
trapallada_data_dir := join(trapallada_dir, 'data')
# conda_environment := "lc"
conda_environment := "trap"

default:
  just --list

# small:
#   #!/usr/bin/env bash
#   source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
#   conda activate {{conda_environment}}
#   # tmux new -s trap trap_fit.py "weights_z=2_epochs=5000.yaml" small

papermill FILE:
  cd "nbs" && papermill {{FILE}} /tmp/{{file_name(FILE)}}

test: (papermill "plot.ipynb") (papermill "data.ipynb") (papermill "xsa_util.ipynb")

# foo:
#   for file in `ls nbs/*.ipynb`; do papermill $file; done

cuda:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  nvidia-smi
  python -c "import jax; jax.devices()"

fit PARAMETERS:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_fit_vae_on_signals.py {{PARAMETERS}} "{{trapallada_data_dir}}"/4_classes_modulated_signals.pt "{{trapallada_data_dir}}"/4_classes_modulated_labels.parquet

download DIR="~/data/trapallada/":
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  trap_download_obs.py {{DIR}}