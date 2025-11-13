# i) PYTHONBREAKPOINT=ipdb.set_trace indicates Python to use ipdb for debugging
# ii) --no-sync doesn't try to update venv before running the command (useuful when having a CUDA-enabled
#     environment in one machine and a CPU-relying in another)
uv_command := "PYTHONBREAKPOINT=ipdb.set_trace uv run --no-sync"

notebooks_dir := join('src', 'imps')

marimo_server := "0.0.0.0"
marimo_port := "2719"

default:
  just -f "{{source_file()}}" --list

# ===========================================

analysis:
  {{uv_command}} marimo edit experiments/analysis.py

selection:
  {{uv_command}} marimo edit experiments/selection.py

assessment:
  {{uv_command}} marimo edit experiments/assessment.py

serve_assessment:
  systemd-run --user --scope tmux new -s assesment '{{uv_command}} marimo run experiments/assessment.py --headless --no-token --host {{marimo_server}} --port {{marimo_port}}'

run_assessment:
  {{uv_command}} marimo run experiments/assessment.py

vae_vs_ar:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py

paper_main_vae_vs_ar:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main"

paper_main_vae_vs_ar_zoomed_east:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "east" --x-range "[90,112]" --y-range "[15,43]"

paper_main_vae_vs_ar_zoomed_north:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "north" --x-range "[-17,17]" --y-range "[85,110]"

paper_main_vae_vs_ar_zoomed_west:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "west" --x-range "[-122,-98]" --y-range "[-6,25]"

paper_main_vae_vs_ar_zoomed_south:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "south" --x-range "[-16,18]" --y-range "[-107,-76]"

paper_main_vae_vs_ar_zoomed_center:
  {{uv_command}} marimo edit experiments/vae_vs_ar.py -- --results_subdir "paper_main" --output_subdir "center" --x-range "[-10,10]" --y-range "[-10,10]"