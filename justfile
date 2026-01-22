mod python '~/scripts/python.just'
mod notebook '~/scripts/notebook.just'
mod mlflow '~/scripts/mlflow.just'
mod git '~/scripts/git.just'

mod latex
mod marimo

home_dir := env_var('HOME')
google_drive_dir := join(home_dir, 'drive')
tmp_dir := '/tmp/'

trapallada_dir := justfile_directory()
trapallada_data_dir := join(trapallada_dir, 'data')

# annotations
reference_signals_file := join(trapallada_dir, 'interesting.yaml')
annotated_windows_file := join(trapallada_dir, 'annotations', 'annotated_windows.parquet')

data_dir := join(env_var_or_default('MY_DATA_DIR', join(home_dir, 'data')), 'trapallada')

# i) PYTHONBREAKPOINT=ipdb.set_trace indicates Python to use ipdb for debugging
# ii) --no-sync doesn't try to update venv before running the command (useuful when having a CUDA-enabled
#     environment in one machine and a CPU-relying in another)
uv_command := "PYTHONBREAKPOINT=ipdb.set_trace uv run --no-sync"

# default_server := "meiga05"
default_server := "skoll"

default:
  just --list

[private]
papermill FILE:
  cd "nbs" && uv run papermill {{FILE}} /tmp/{{file_name(FILE)}}

test: (papermill "plot.ipynb") (papermill "xsa_util.ipynb") (papermill "xsa_access.ipynb") (papermill "util.ipynb") (papermill "model_nn.ipynb") (papermill "model_ar.ipynb") (papermill "results_analysis.ipynb") (papermill "data.ipynb")

# ====================================================================================================

[private]
select_gpu_and_fit SCRIPT PARAMETERS DATA_FILE MASKS_FILE METADATA_FILE DATA_HYPERPARAMETERS_FILE="" ANNOTATED_WINDOWS_TO_VALIDATION="true":
  #!/usr/bin/env bash

  export CUDA_VISIBLE_DEVICES=$(select-gpu.sh)
  echo Selecting GPU \# $CUDA_VISIBLE_DEVICES
  if [ "$CUDA_VISIBLE_DEVICES" -eq -1 ]; then
    echo "⚠️ nvidia-smi not found. CUDA_VISIBLE_DEVICES not set" >&2
  fi

  {{ if ANNOTATED_WINDOWS_TO_VALIDATION == "true" { "echo ❗ Using annotated windows and reference signals files to assemble the validation set" } else { "" } }}

  {{uv_command}} {{SCRIPT}} {{PARAMETERS}} "{{trapallada_data_dir}}"/"{{DATA_FILE}}" "{{trapallada_data_dir}}"/"{{MASKS_FILE}}" "{{trapallada_data_dir}}"/"{{METADATA_FILE}}" "{{trapallada_data_dir}}"/"{{DATA_HYPERPARAMETERS_FILE}}" {{ if ANNOTATED_WINDOWS_TO_VALIDATION == "true" { "--reference-signals-file " +  reference_signals_file + " --annotated-windows-file " + annotated_windows_file } else { "" } }}

[group('training')]
[doc("PARAMETERS: a YAML file; DATA_FILES_DIR: the *sub*-directory (of the predetermined data directory) where .pt and .parquet files are located")]
fit PARAMETERS DATA_FILES_DIR  ANNOTATED_WINDOWS_TO_VALIDATION="true": (select_gpu_and_fit "trap_fit_vae_on_signals.py"  PARAMETERS DATA_FILES_DIR + "/signals.pt" DATA_FILES_DIR + "/masks.pt" DATA_FILES_DIR + "/metadata.parquet" DATA_FILES_DIR + "/data_preprocessing_parameters.yaml" ANNOTATED_WINDOWS_TO_VALIDATION)

[group('training')]
[doc("PARAMETERS: a YAML file; DATA_FILES_DIR: the *sub*-directory (of the predetermined data directory) where .pt and .parquet files are located")]
grid_fit PARAMETERS DATA_FILES_DIR  ANNOTATED_WINDOWS_TO_VALIDATION="true": (select_gpu_and_fit "trap_grid_fit_vae_on_signals.py" PARAMETERS DATA_FILES_DIR + "/signals.pt" DATA_FILES_DIR + "/masks.pt" DATA_FILES_DIR + "/metadata.parquet" DATA_FILES_DIR + "/data_preprocessing_parameters.yaml" ANNOTATED_WINDOWS_TO_VALIDATION)

[group('training')]
[doc("PARAMETERS: a YAML file")]
fit_test PARAMETERS="parameters.yaml"  ANNOTATED_WINDOWS_TO_VALIDATION="true": && (fit tmp_dir/PARAMETERS "mini_rate" ANNOTATED_WINDOWS_TO_VALIDATION)
  cp {{PARAMETERS}} {{tmp_dir}}

[group('training')]
[doc("PARAMETERS: a YAML file")]
grid_fit_test PARAMETERS="parameters.yaml" ANNOTATED_WINDOWS_TO_VALIDATION="true": && (grid_fit tmp_dir/PARAMETERS "mini_rate" ANNOTATED_WINDOWS_TO_VALIDATION)
  cp {{PARAMETERS}} {{tmp_dir}}

# [group('training')]
# fit_test_fft PARAMETERS="parameters.yaml": && (fit tmp_dir/"fft_" + PARAMETERS "mini_rate")
#   cp {{PARAMETERS}} {{tmp_dir}}/"fft_"{{PARAMETERS}}
#   patch {{tmp_dir}}/"fft_"{{PARAMETERS}} < "samples/fft.patch"

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
pack_obs_test OUTPUT_SUBDIR="mini_rate": (pack_obs OUTPUT_SUBDIR "63" "225" "10" "4" "0" "0.02" "0.2" data_dir/"sample" trapallada_data_dir)

# pack_obs_min_count_rate_gti_ratio OUTPUT_SUBDIR MIN_COUNT_RATE="0.02" MIN_GTI_RATIO="0.2": (pack_obs OUTPUT_SUBDIR "63" "225" "10" "4" "0" MIN_COUNT_RATE MIN_GTI_RATIO data_dir trapallada_data_dir)

whole_sheebang PARAMETERS DATA_FILES_STEM="obs": (download_obs) (pack_obs DATA_FILES_STEM) (fit PARAMETERS DATA_FILES_STEM)
  echo "this might take a while..."

# ====================================================================================================

[group('overleaf')]
paper:
  xdg-open "https://www.overleaf.com/project/68da64d6929dc7af28c157b2"