home_dir := env_var('HOME')
google_drive_dir := join(home_dir, 'drive')
tmp_dir := '/tmp/'

# light_curves_dir := join(home_dir, 'Sync/research/light curves')

trapallada_dir := justfile_directory()
trapallada_data_dir := join(trapallada_dir, 'data')

# data_dir := join(env_var_or_default('MY_DATA_DIR', join(home_dir, 'data')), 'trapallada')
# data_dir := join(env_var_or_default('MY_DATA_DIR', join(home_dir, 'data')), 'trapallada2')
data_dir := join(env_var_or_default('MY_DATA_DIR', join(home_dir, 'data')), 'trapallada3')

conda_environment := "trap"

# i) PYTHONBREAKPOINT=ipdb.set_trace indicates Python to use ipdb for debugging
# ii) --no-sync doesn't try to update venv before running the command (useuful when having a CUDA-enabled
#     environment in one machine and a CPU-relying in another)
uv_command := "PYTHONBREAKPOINT=ipdb.set_trace uv run --no-sync"

# default_server := "meiga05"
default_server := "skoll"

marimo_server := "0.0.0.0"
marimo_port := "2719"

import '~/python/import.just'

default:
  just --list

[private]
papermill FILE:
  cd "nbs" && uv run papermill {{FILE}} /tmp/{{file_name(FILE)}}

test: (papermill "plot.ipynb") (papermill "xsa_util.ipynb") (papermill "xsa_access.ipynb") (papermill "util.ipynb") (papermill "model_nn.ipynb") (papermill "model_ar.ipynb") (papermill "results_analysis.ipynb") (papermill "data.ipynb")

# ========================================analysis ===================================================

[group('analysis')]
[doc("NOTEBOOK: the name, without extension, of a notebook in experiments directory; DIR: the path to a results directory (containing the eqx, yaml, and parquet files)")]
run_parameterized_notebook NOTEBOOK DIR:
  {{uv_command}} papermill experiments/{{NOTEBOOK}}.ipynb {{parent_directory(DIR)}}/{{NOTEBOOK}}_{{file_stem(DIR)}}.ipynb -p model_subdir {{DIR}}

[group('analysis')]
[doc("MODEL_DIR: the path to a results directory (containing the eqx, yaml, and parquet files), relative to `trapallada`; GRAPH_DIR: the path to the directory with the graph files (nodes, edges), either absolute or relative to the Google Drive directory; SUFFIX: optional suffix to append to the output file")]
run_graphs_notebook MODEL_DIR GRAPH_DIR SUFFIX="":
  {{uv_command}} papermill experiments/graphs_post_processing.ipynb {{parent_directory(MODEL_DIR)}}/graphs_post_processing_{{file_stem(MODEL_DIR)}}{{SUFFIX}}.ipynb -p model_subdir {{MODEL_DIR}} -p graph_subdir {{GRAPH_DIR}}

[group('analysis')]
[doc("MODEL_DIR: *absolute* path to a results directory (containing the eqx, yaml, and parquet files); GRAPH_DIR: *absolute* path to the directory with the graph files (nodes, edges); DATA_DIR: the path to the directory with the data; SUFFIX: optional suffix to append to the output file")]
run_validation_notebook MODEL_DIR GRAPH_DIR SUFFIX="":
  {{uv_command}} papermill experiments/validation.ipynb {{parent_directory(MODEL_DIR)}}/validation_{{file_stem(MODEL_DIR)}}{{SUFFIX}}.ipynb -p model_dir {{MODEL_DIR}} -p graph_dir {{GRAPH_DIR}}

# ====================================================================================================

[group('results')]
analysis:
  {{uv_command}} marimo edit experiments/analysis.py

[group('results')]
selection:
  {{uv_command}} marimo edit experiments/selection.py

[group('results')]
assessment:
  {{uv_command}} marimo edit experiments/assessment.py

# [group('results')]
# validation:
#   {{uv_command}} marimo edit experiments/validation.py

[group('results')]
serve_assessment:
  systemd-run --user --scope tmux new -s assesment '{{uv_command}} marimo run experiments/assessment.py --headless --no-token --host {{marimo_server}} --port {{marimo_port}}'

[group('results')]
run_assessment:
  {{uv_command}} marimo run experiments/assessment.py

# ====================================================================================================

[private]
select_gpu_and_fit PARAMETERS DATA_FILE MASKS_FILE METADATA_FILE DATA_HYPERPARAMETERS_FILE="":
  #!/usr/bin/env bash

  export CUDA_VISIBLE_DEVICES=$(select-gpu.sh)
  # echo $CUDA_VISIBLE_DEVICES
  if [ "$CUDA_VISIBLE_DEVICES" -eq -1 ]; then
    echo "⚠️ nvidia-smi not found. CUDA_VISIBLE_DEVICES not set" >&2
  fi
  
  {{uv_command}} trap_fit_vae_on_signals.py {{PARAMETERS}} "{{trapallada_data_dir}}"/"{{DATA_FILE}}" "{{trapallada_data_dir}}"/"{{MASKS_FILE}}" "{{trapallada_data_dir}}"/"{{METADATA_FILE}}" "{{trapallada_data_dir}}"/"{{DATA_HYPERPARAMETERS_FILE}}"

[group('training')]
[doc("PARAMETERS: a YAML file; DATA_FILES_DIR: the *sub*-directory (of the predetermined data directory) where .pt and .parquet files are located")]
fit PARAMETERS DATA_FILES_DIR: (select_gpu_and_fit PARAMETERS DATA_FILES_DIR + "/signals.pt" DATA_FILES_DIR + "/masks.pt" DATA_FILES_DIR + "/metadata.parquet" DATA_FILES_DIR + "/data_preprocessing_parameters.yaml")

[group('training')]
[doc("PARAMETERS: a YAML file")]
fit_test PARAMETERS="parameters.yaml": && (fit tmp_dir/PARAMETERS "mini_rate")
  cp {{PARAMETERS}} {{tmp_dir}}

[group('training')]
fit_test_fft PARAMETERS="parameters.yaml": && (fit tmp_dir/"fft_" + PARAMETERS "mini_rate")
  cp {{PARAMETERS}} {{tmp_dir}}/"fft_"{{PARAMETERS}}
  patch {{tmp_dir}}/"fft_"{{PARAMETERS}} < "samples/fft.patch"

# ========================================= data =====================================================

[group('data')]
[doc("downloads tar files in DOWNLOAD_DIR")]
download_obs DIR=data_dir MIN_DURATION="35000" MAX_DURATION="60000":
  mkdir -p {{DIR}}
  {{uv_command}} trap_download_obs.py {{DIR}} --min-duration {{MIN_DURATION}} --max-duration {{MAX_DURATION}}

[group('data')]
[doc("packs previously downloaded tar files located in DOWNLOAD_DIR into a format fit for training; resulting files ared placed in OUTPUT_DIR/OUTPUT_SUBDIR")]
pack_obs OUTPUT_SUBDIR BIN_SIZE="63" WINDOW_SIZE="225" STEP="10" MAX_N_SOURCES="4" MIN_POWER="0" MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2" DOWNLOAD_DIR=data_dir OUTPUT_DIR=trapallada_data_dir +SIGNALS="RATE1 RATE2 RATE3 RATE4 RATE5":
  {{uv_command}} trap_pack_obs.py {{DOWNLOAD_DIR}} {{OUTPUT_DIR}} {{OUTPUT_SUBDIR}} -b {{BIN_SIZE}} -w {{WINDOW_SIZE}} -s {{STEP}} -M {{MAX_N_SOURCES}} -m {{MIN_POWER}} -c {{MIN_COUNT_RATE}} -g {{MIN_GTI_RATIO}} -S {{SIGNALS}}
  cp "logs/trapallada.xsa.util.log" {{OUTPUT_DIR}}/{{OUTPUT_SUBDIR}}

[group('data')]
pack_obs_with_background OUTPUT_SUBDIR BIN_SIZE="63" WINDOW_SIZE="225" STEP="10" MAX_N_SOURCES="4" MIN_POWER="0" MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2" DOWNLOAD_DIR=data_dir OUTPUT_DIR=trapallada_data_dir: (pack_obs OUTPUT_SUBDIR BIN_SIZE WINDOW_SIZE STEP MAX_N_SOURCES MIN_POWER MIN_COUNT_RATE MIN_GTI_RATIO DOWNLOAD_DIR OUTPUT_DIR "RATE BACKV")

[group('data')]
pack_raw_obs OUTPUT_SUBDIR: (pack_obs OUTPUT_SUBDIR "1" "14175" "630")

[group('data')]
pack_obs_test OUTPUT_SUBDIR="mini_rate": (pack_obs OUTPUT_SUBDIR "63" "225" "10" "4" "0" "0" "0" data_dir/"sample" trapallada_data_dir)

# pack_obs_min_count_rate_gti_ratio OUTPUT_SUBDIR MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2": (pack_obs OUTPUT_SUBDIR "63" "225" "10" "4" "0" MIN_COUNT_RATE MIN_GTI_RATIO data_dir trapallada_data_dir)

whole_sheebang PARAMETERS DATA_FILES_STEM="obs": (download_obs) (pack_obs DATA_FILES_STEM) (fit PARAMETERS DATA_FILES_STEM)
  echo "this might take a while..."

# ====================================================================================================

[group('infracstructure')]
retrieve_from_cluster SERVER=default_server DIRECTORY="results":
  rsync --recursive {{SERVER}}:"trapallada"/{{DIRECTORY}} {{tmp_dir}}/
  @echo "Retrieving all the files in {{SERVER}} at {{DIRECTORY}}"

[group('infracstructure')]
mlflow_ui:
  echo {{env_var('MLFLOW_TRACKING_URI')}}
  {{uv_command}} mlflow ui --port 8080 --backend-store-uri {{env_var('MLFLOW_TRACKING_URI')}}

[group('infracstructure')]
mlflow_server:
  {{uv_command}} mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri {{env_var('MLFLOW_TRACKING_URI')}}

[group('infracstructure')]
connect_to_remote_mlflow_ui SERVER=default_server BROWSER="falkon":
  #!/usr/bin/env bash
  ssh -N -L 10000:localhost:8080 {{SERVER}} &
  SSH_PID=$!
  {{BROWSER}} http://localhost:10000
  kill $SSH_PID

[doc("FILE: the name of the html file; NAME: new name for the file; SERVER: remote server where the notebook will be published")]
[group('infracstructure')]
publish_html_on_cluster FILE NAME SERVER=default_server:
  cp "{{FILE}}" {{tmp_dir}}/{{NAME}}.html
  rsync --perms --owner {{tmp_dir}}/{{NAME}}.html {{SERVER}}:public_html
  rm {{tmp_dir}}/{{NAME}}.html
  echo https://www.tsc.uc3m.es/~mvazquez/{{NAME}}.html

clean_cache DIR=".":
  # rm experiments/*.parquet
  fd -u -g "{nodes,edges}.parquet" --exec rm {} \; {{DIR}}

# ====================================================================================================

[group('jesus')]
mount_supergraph:
  rclone mount --read-only uc3m: {{google_drive_dir}}/git_supergraph --drive-root-folder-id 1czYxj2lzIyVCHQj1uVefYz6L7OWV_iY5

[group('jesus')]
mount_trapallada:
  rclone mount uc3m: {{google_drive_dir}}/trapallada --drive-root-folder-id 1nyxuRvEFIsEnAXVP0Z0wxJtJRLOPWB7R

[group('jesus')]
cp_results INPUT_DIR:
  #!/usr/bin/env bash
  set -euo pipefail
  
  output_dir={{google_drive_dir}}/trapallada/{{datetime('%Y-%m-%d-%H%M%S')}}
  echo Copying to $output_dir...
  mkdir $output_dir
  echo "{{INPUT_DIR}}" > $output_dir/source.txt
  # VAE
  cp {{INPUT_DIR}}/vae.parquet $output_dir
  # AR
  cp $(cat {{INPUT_DIR}}/signals_to_assess.txt) $output_dir
  cp {{INPUT_DIR}}/labels/annotated_windows.parquet $output_dir
