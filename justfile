home_dir := env_var('HOME')
google_drive_dir := join(home_dir, 'drive')
tmp_dir := '/tmp/'

light_curves_dir := join(home_dir, 'Sync/research/light curves')

trapallada_dir := join(home_dir, 'Sync/git/trapallada')
trapallada_data_dir := join(trapallada_dir, 'data')

data_dir := join(env_var_or_default('MY_DATA_DIR', join(home_dir, 'data')), 'trapallada')

conda_environment := "trap"

import '~/python/import.just'

default:
  just --list

[private]
papermill FILE:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  cd "nbs" && papermill {{FILE}} /tmp/{{file_name(FILE)}}

# test: (papermill "plot.ipynb") (papermill "data.ipynb") (papermill "xsa_util.ipynb") (papermill "xsa_access.ipynb") (papermill "util.ipynb") (papermill "model_signal.ipynb")
test: (papermill "plot.ipynb") (papermill "xsa_util.ipynb") (papermill "xsa_access.ipynb") (papermill "util.ipynb") (papermill "model_signal.ipynb")

# run_results_notebook STEM:
#   #!/usr/bin/env bash
#   source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
#   conda activate {{conda_environment}}
#   papermill experiments/results.ipynb {{parent_directory(STEM)}}/{{file_stem(STEM)}}.ipynb -p input_files_stem {{STEM}}
#   # echo papermill {{join(trapallada_dir, 'experiments', 'results.ipynb')}} {{join(invocation_directory(), parent_directory(STEM), file_stem(STEM) + ".ipynb")}} -p input_files_stem {{STEM}}

[doc("NOTEBOOK: the name, without extension, of a notebook in experiments directory; DIR: the path to a results directory (containing the eqx, yaml, and parquet files)")]
run_parameterized_notebook NOTEBOOK DIR:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  papermill experiments/{{NOTEBOOK}}.ipynb {{parent_directory(DIR)}}/{{NOTEBOOK}}_{{file_stem(DIR)}}.ipynb -p model_subdir {{DIR}}

cuda:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  nvidia-smi
  python -c "import jax; print(jax.devices())"

[private]
fit PARAMETERS DATA_FILE METADATA_FILE DATA_HYPERPARAMETERS_FILE="":
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
  
  trap_fit_vae_on_signals.py {{PARAMETERS}} "{{trapallada_data_dir}}"/"{{DATA_FILE}}" "{{trapallada_data_dir}}"/"{{METADATA_FILE}}" "{{trapallada_data_dir}}"/"{{DATA_HYPERPARAMETERS_FILE}}"

fit_synthetic PARAMETERS: (fit PARAMETERS "4_classes_modulated_signals.pt" "4_classes_modulated_labels.parquet")

fit_real PARAMETERS DATA_FILES_DIR: (fit PARAMETERS DATA_FILES_DIR + "/signals.pt" DATA_FILES_DIR + "/metadata.parquet" DATA_FILES_DIR + "/data_preprocessing_parameters.yaml")

fit_test PARAMETERS="vanilla_mini.yaml": (fit_real "results"/PARAMETERS "mini_rate_background")

download_obs DIR=data_dir:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_download_obs.py {{DIR}}

pack_obs OUTPUT_SUBDIR BIN_SIZE="63" WINDOW_SIZE="225" STEP="10" MAX_N_SOURCES="4" MIN_POWER="0" DOWNLOAD_DIR=data_dir OUTPUT_DIR=trapallada_data_dir:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_pack_obs.py {{DOWNLOAD_DIR}} {{OUTPUT_DIR}} {{OUTPUT_SUBDIR}} -b {{BIN_SIZE}} -w {{WINDOW_SIZE}} -s {{STEP}} -M {{MAX_N_SOURCES}} -m {{MIN_POWER}}
  # cp "logs/trapallada.xsa.util.log" {{OUTPUT_DIR}}/{{OUTPUT_SUBDIR}}.log
  cp "logs/trapallada.xsa.util.log" {{OUTPUT_DIR}}/{{OUTPUT_SUBDIR}}

pack_obs_test: (pack_obs "mini_rate_background" "63" "225" "10" "4" "0" data_dir/"sample" trapallada_data_dir)

whole_sheebang PARAMETERS DATA_FILES_STEM="obs": (download_obs) (pack_obs DATA_FILES_STEM) (fit_real PARAMETERS DATA_FILES_STEM)
  echo "this might take a while..."

retrieve_from_cluster SERVER="meiga04" DIRECTORY="trapallada/results":
  rsync --recursive {{SERVER}}:{{DIRECTORY}} {{tmp_dir}}/
  @echo "Retrieving all the files in {{SERVER}} at {{DIRECTORY}}"

mlflow_ui:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  echo {{env_var('MLFLOW_TRACKING_URI')}}
  mlflow ui --port 8080 --backend-store-uri {{env_var('MLFLOW_TRACKING_URI')}}

mlflow_server:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri {{env_var('MLFLOW_TRACKING_URI')}}

connect_to_remote_mlflow_ui SERVER="meiga04" BROWSER="falkon":
  #!/usr/bin/env bash
  ssh -N -L 10000:localhost:8080 {{SERVER}} &
  SSH_PID=$!
  {{BROWSER}} http://localhost:10000
  kill $SSH_PID

[doc("FILE: the name of the notebook; SERVER: remote server where the notebook will be published")]
publish_on_cluster FILE SERVER="meiga04": (export_notebook_to_html FILE)
  chmod go+r {{without_extension(FILE)}}.html
  rsync --perms --owner {{without_extension(FILE)}}.html {{SERVER}}:public_html
  rm {{without_extension(FILE)}}.html
  echo https://www.tsc.uc3m.es/~mvazquez/{{file_stem(FILE)}}.html

clean_cache DIR=".":
  # rm experiments/*.parquet
  fd -u -g "{nodes,edges}.parquet" --exec rm {} \; {{DIR}}

mount_supergraph:
  rclone mount uc3m: {{google_drive_dir}}/git_supergraph --drive-root-folder-id 1czYxj2lzIyVCHQj1uVefYz6L7OWV_iY5

mount_trapallada:
  rclone mount uc3m: {{google_drive_dir}}/trapallada --drive-root-folder-id 1nyxuRvEFIsEnAXVP0Z0wxJtJRLOPWB7R

  