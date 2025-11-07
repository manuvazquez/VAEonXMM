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