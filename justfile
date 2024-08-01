home_dir := env_var('HOME')

light_curves_dir := join(home_dir, 'Sync/research/light curves')

trapallada_dir := join(home_dir, 'Sync/git/trapallada')
trapallada_data_dir := join(trapallada_dir, 'data')

# data_dir := join(env_var('MY_DATA_DIR'), 'trapallada')
data_dir := join(env_var_or_default('MY_DATA_DIR', join(home_dir, 'data')), 'trapallada')

conda_environment := "trap"

default:
  just --list

[private]
papermill FILE:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  cd "nbs" && papermill {{FILE}} /tmp/{{file_name(FILE)}}

test: (papermill "plot.ipynb") (papermill "data.ipynb") (papermill "xsa_util.ipynb")

cuda:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  nvidia-smi
  python -c "import jax; print(jax.devices())"

[private]
fit PARAMETERS DATA_FILE METADATA_FILE:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  
  # check if nvidia-smi is available
  if command -v nvidia-smi &> /dev/null; then
      
      # the GPU with the most free memory
      BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '{print $1, $2}' | sort -k2 -nr | head -n1 | awk '{print $1}')
      
      echo "Best GPU index: $BEST_GPU"

      export CUDA_VISIBLE_DEVICES=$BEST_GPU
  else
      echo "nvidia-smi not found. CUDA_VISIBLE_DEVICES not set."
  fi
  
  trap_fit_vae_on_signals.py {{PARAMETERS}} "{{trapallada_data_dir}}"/"{{DATA_FILE}}" "{{trapallada_data_dir}}"/"{{METADATA_FILE}}"

fit_synthetic PARAMETERS: (fit PARAMETERS "4_classes_modulated_signals.pt" "4_classes_modulated_labels.parquet")

fit_real PARAMETERS DATA_FILES_STEM: (fit PARAMETERS DATA_FILES_STEM + "_signals.pt" DATA_FILES_STEM + "_metadata.parquet")

download_obs DIR=data_dir:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_download_obs.py {{DIR}}

pack_obs OUTPUT_FILES_STEM BIN_SIZE="20" WINDOW_SIZE="600" MIN_POWER="0.5" DOWNLOAD_DIR=data_dir OUTPUT_DIR=trapallada_data_dir:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_pack_obs.py {{DOWNLOAD_DIR}} {{OUTPUT_DIR}} {{OUTPUT_FILES_STEM}} -b {{BIN_SIZE}} -w {{WINDOW_SIZE}} -p {{MIN_POWER}}

whole_sheebang PARAMETERS DATA_FILES_STEM="obs": (download_obs) (pack_obs DATA_FILES_STEM) (fit_real PARAMETERS DATA_FILES_STEM)
  echo "this might take a while..."