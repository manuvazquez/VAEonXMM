home_dir := env_var('HOME')
google_drive_dir := join(home_dir, 'drive')
tmp_dir := '/tmp/'

# light_curves_dir := join(home_dir, 'Sync/research/light curves')

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

test: (papermill "plot.ipynb") (papermill "xsa_util.ipynb") (papermill "xsa_access.ipynb") (papermill "util.ipynb") (papermill "model_nn.ipynb")

[doc("NOTEBOOK: the name, without extension, of a notebook in experiments directory; DIR: the path to a results directory (containing the eqx, yaml, and parquet files)")]
run_parameterized_notebook NOTEBOOK DIR:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  papermill experiments/{{NOTEBOOK}}.ipynb {{parent_directory(DIR)}}/{{NOTEBOOK}}_{{file_stem(DIR)}}.ipynb -p model_subdir {{DIR}}

run_analysis DIR: (run_parameterized_notebook "results" DIR) (run_parameterized_notebook "analysis" DIR)

[doc("MODEL_DIR: the path to a results directory (containing the eqx, yaml, and parquet files), relative to `trapallada`; GRAPH_DIR: the path to the directory with the graph files (nodes, edges), either absolute or relative to the Google Drive directory; SUFFIX: optional suffix to append to the output file")]
run_graphs_notebook MODEL_DIR GRAPH_DIR SUFFIX="":
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  papermill experiments/graphs_post_processing.ipynb {{parent_directory(MODEL_DIR)}}/graphs_post_processing_{{file_stem(MODEL_DIR)}}{{SUFFIX}}.ipynb -p model_subdir {{MODEL_DIR}} -p graph_subdir {{GRAPH_DIR}}

[doc("MODEL_DIR: *absolute* path to a results directory (containing the eqx, yaml, and parquet files); GRAPH_DIR: *absolute* path to the directory with the graph files (nodes, edges); DATA_DIR: the path to the directory with the data; SUFFIX: optional suffix to append to the output file")]
run_validation_notebook MODEL_DIR GRAPH_DIR SUFFIX="":
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  papermill experiments/validation.ipynb {{parent_directory(MODEL_DIR)}}/validation_{{file_stem(MODEL_DIR)}}{{SUFFIX}}.ipynb -p model_dir {{MODEL_DIR}} -p graph_dir {{GRAPH_DIR}}

cuda:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  nvidia-smi
  python -c "import jax; print(jax.devices())"

[private]
fit PARAMETERS DATA_FILE MASKS_FILE METADATA_FILE DATA_HYPERPARAMETERS_FILE="":
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
  
  trap_fit_vae_on_signals.py {{PARAMETERS}} "{{trapallada_data_dir}}"/"{{DATA_FILE}}" "{{trapallada_data_dir}}"/"{{MASKS_FILE}}" "{{trapallada_data_dir}}"/"{{METADATA_FILE}}" "{{trapallada_data_dir}}"/"{{DATA_HYPERPARAMETERS_FILE}}"

fit_real PARAMETERS DATA_FILES_DIR: (fit PARAMETERS DATA_FILES_DIR + "/signals.pt" DATA_FILES_DIR + "/masks.pt" DATA_FILES_DIR + "/metadata.parquet" DATA_FILES_DIR + "/data_preprocessing_parameters.yaml")

[doc("PARAMETERS: a YAML file")]
fit_test PARAMETERS="parameters.yaml": && (fit_real tmp_dir/PARAMETERS "mini_rate")
  cp {{PARAMETERS}} {{tmp_dir}}

fit_test_fft PARAMETERS="parameters.yaml": && (fit_real tmp_dir/"fft_" + PARAMETERS "mini_rate")
  cp {{PARAMETERS}} {{tmp_dir}}/"fft_"{{PARAMETERS}}
  patch {{tmp_dir}}/"fft_"{{PARAMETERS}} < "samples/fft.patch"

download_obs DIR=data_dir:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_download_obs.py {{DIR}}

pack_obs OUTPUT_SUBDIR BIN_SIZE="63" WINDOW_SIZE="225" STEP="10" MAX_N_SOURCES="4" MIN_POWER="0" MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2" DOWNLOAD_DIR=data_dir OUTPUT_DIR=trapallada_data_dir +SIGNALS="RATE":
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  trap_pack_obs.py {{DOWNLOAD_DIR}} {{OUTPUT_DIR}} {{OUTPUT_SUBDIR}} -b {{BIN_SIZE}} -w {{WINDOW_SIZE}} -s {{STEP}} -M {{MAX_N_SOURCES}} -m {{MIN_POWER}} -c {{MIN_COUNT_RATE}} -g {{MIN_GTI_RATIO}} -S {{SIGNALS}}
  cp "logs/trapallada.xsa.util.log" {{OUTPUT_DIR}}/{{OUTPUT_SUBDIR}}

pack_obs_with_background OUTPUT_SUBDIR BIN_SIZE="63" WINDOW_SIZE="225" STEP="10" MAX_N_SOURCES="4" MIN_POWER="0" MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2" DOWNLOAD_DIR=data_dir OUTPUT_DIR=trapallada_data_dir: (pack_obs OUTPUT_SUBDIR BIN_SIZE WINDOW_SIZE STEP MAX_N_SOURCES MIN_POWER MIN_COUNT_RATE MIN_GTI_RATIO DOWNLOAD_DIR OUTPUT_DIR "RATE BACKV")

pack_raw_obs OUTPUT_SUBDIR: (pack_obs OUTPUT_SUBDIR "1" "14175" "630")

pack_obs_test OUTPUT_SUBDIR="mini_rate": (pack_obs OUTPUT_SUBDIR "63" "225" "10" "4" "0" "0" "0" data_dir/"sample" trapallada_data_dir)

# pack_obs_min_count_rate_gti_ratio OUTPUT_SUBDIR MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2": (pack_obs OUTPUT_SUBDIR "63" "225" "10" "4" "0" MIN_COUNT_RATE MIN_GTI_RATIO data_dir trapallada_data_dir)

whole_sheebang PARAMETERS DATA_FILES_STEM="obs": (download_obs) (pack_obs DATA_FILES_STEM) (fit_real PARAMETERS DATA_FILES_STEM)
  echo "this might take a while..."

retrieve_from_cluster SERVER="meiga04" DIRECTORY="results":
  rsync --recursive {{SERVER}}:"trapallada"/{{DIRECTORY}} {{tmp_dir}}/
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
  rclone mount --read-only uc3m: {{google_drive_dir}}/git_supergraph --drive-root-folder-id 1czYxj2lzIyVCHQj1uVefYz6L7OWV_iY5

mount_trapallada:
  rclone mount uc3m: {{google_drive_dir}}/trapallada --drive-root-folder-id 1nyxuRvEFIsEnAXVP0Z0wxJtJRLOPWB7R

  