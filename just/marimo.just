# i) PYTHONBREAKPOINT=ipdb.set_trace indicates Python to use ipdb for debugging
# ii) --no-sync doesn't try to update venv before running the command (useuful when having a CUDA-enabled
#     environment in one machine and a CPU-relying in another)
uv_command := "PYTHONBREAKPOINT=ipdb.set_trace uv run --no-sync"

notebooks_dir := join(justfile_directory(), 'experiments')

marimo_server := "0.0.0.0"
marimo_port := "2719"

default:
  just -f "{{source_file()}}" --list

# ===========================================

[no-cd]
analysis:
  {{uv_command}} marimo edit experiments/analysis.py

[no-cd]
vae_analysis DIR:
  {{uv_command}} python {{join(notebooks_dir,"analysis.py")}} -- --algorithm "vae" --model_dir {{DIR}}

[no-cd]
selection:
  {{uv_command}} marimo edit experiments/selection.py

[no-cd]
assessment:
  {{uv_command}} marimo edit experiments/assessment.py

[no-cd]
serve_assessment:
  systemd-run --user --scope tmux new -s assesment '{{uv_command}} marimo run experiments/assessment.py --headless --no-token --host {{marimo_server}} --port {{marimo_port}}'

[no-cd]
run_assessment:
  {{uv_command}} marimo run experiments/assessment.py

[no-cd]
vae_vs_ar:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py

[no-cd]
paper_main_vae_vs_ar:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main"

[no-cd]
paper_main_vae_vs_ar_zoomed_east:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "east" --x-range "[90,112]" --y-range "[15,43]"

[no-cd]
paper_main_vae_vs_ar_zoomed_north:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "north" --x-range "[-17,17]" --y-range "[85,110]"

[no-cd]
paper_main_vae_vs_ar_zoomed_west:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "west" --x-range "[-122,-98]" --y-range "[-6,25]"

[no-cd]
paper_main_vae_vs_ar_zoomed_south:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "south" --x-range "[-16,18]" --y-range "[-107,-76]"

[no-cd]
paper_main_vae_vs_ar_zoomed_center:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "center" --x-range "[-10,10]" --y-range "[-10,10]"