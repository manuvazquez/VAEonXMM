# set dotenv-load
home_dir := env_var('HOME')
# tmp_dir := '/tmp/'

light_curves_dir := join(home_dir, 'Sync/research/light curves')
conda_environment := "lc"
# for_students_newcommand := "\\newcommand{\\TforStudents}{}"

default:
  just --list

small:
  #!/usr/bin/env bash
  source $HOME/$MY_CONDA_INSTALLATION/etc/profile.d/conda.sh
  conda activate {{conda_environment}}
  # tmux new -s trap trap_fit.py "weights_z=2_epochs=5000.yaml" small

