output_dir := "figures"

latexmk_command := "latexmk -lualatex -quiet -gg"
latexmk_outdir := "-outdir=" + output_dir

default:
  just -f {{source_file()}} --list

all: (reference_vs_non-reference "paper_main" "44-58_91-100") (reference_vs_non-reference "ae_z_10") (embedded_window "paper_main/106-117_-16-2/11099") (validation "paper_main" "-95--84_58-66") (validation "paper_main") (roc "paper_main" "vae") (roc "paper_main" "ar") (latent "paper_main")

dolphin:
  # dolphin --new-window plots/{{output_dir}} &
  dolphin plots/{{output_dir}} &

# ==================================================

[private]
reference_vs_non-reference_allowing_clean DIR ZOOM_SUBDIR="" CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataFile}{"{{DIR}}"/tSNE.csv}"{{if ZOOM_SUBDIR!="" {"\\\\newcommand{\\\\zoomSubdir}{"+ZOOM_SUBDIR+"}"} else {""} }} -jobname=reference_vs_non-reference_{{DIR}} {{latexmk_outdir}} -cd "plots/reference_vs_non-reference.tex"

reference_vs_non-reference DIR ZOOM_SUBDIR="": (reference_vs_non-reference_allowing_clean DIR ZOOM_SUBDIR) (reference_vs_non-reference_allowing_clean DIR ZOOM_SUBDIR "true")

# ==================================================

[private]
embedded_window_allowing_clean DIR CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\subdir}{"{{DIR}}"}" -cd "plots/embedded_window.tex"

embedded_window DIR: (embedded_window_allowing_clean DIR) (embedded_window_allowing_clean DIR "true")
  mv plots/embedded_window.pdf plots/{{output_dir}}/embedded_window_{{replace(DIR, "/", "_")}}.pdf

# ==================================================

[private]
validation_allowing_clean DIR ZOOM_SUBDIR="" CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataDir}{"{{DIR}}"}"{{if ZOOM_SUBDIR!="" {"\\\\newcommand{\\\\zoomSubdir}{"+ZOOM_SUBDIR+"}"} else {""} }} -cd "plots/validation.tex"

validation DIR ZOOM_SUBDIR="": (validation_allowing_clean DIR ZOOM_SUBDIR) (validation_allowing_clean DIR ZOOM_SUBDIR "true")
  mv plots/validation.pdf plots/{{output_dir}}/validation_{{replace(DIR, "/", "_")}}{{if ZOOM_SUBDIR!="" {ZOOM_SUBDIR} else {""} }}.pdf

# ==================================================

[private]
roc_allowing_clean DIR ALGORITHM CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataDir}{"{{DIR}}"}\\newcommand{\\algo}{"{{ALGORITHM}}"}" -cd "plots/roc.tex"

roc DIR ALGORITHM: (roc_allowing_clean DIR ALGORITHM) (roc_allowing_clean DIR ALGORITHM "true")
  mv plots/roc.pdf plots/{{output_dir}}/{{ALGORITHM}}_roc_{{replace(DIR, "/", "_")}}.pdf

# ==================================================

[private]
latent_allowing_clean DIR ZOOM_SUBDIRS="" CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataDir}{"{{DIR}}"}"{{if ZOOM_SUBDIRS!="" {"\\\\newcommand{\\\\zoomSubdirs}{"+ZOOM_SUBDIRS+"}"} else {""} }} -cd "plots/latent.tex"

latent DIR ZOOM_SUBDIRS="": (latent_allowing_clean DIR ZOOM_SUBDIRS) (latent_allowing_clean DIR ZOOM_SUBDIRS "true")
  mv plots/latent.pdf plots/{{output_dir}}/latent_{{replace(DIR, "/", "_")}}{{if ZOOM_SUBDIRS!="" {"_"+ZOOM_SUBDIRS} else {""} }}.pdf

# ==================================================

[private]
window_allowing_clean DIR CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dir}{"{{DIR}}"}" -cd "plots/window.tex"

window DIR: (window_allowing_clean DIR) (window_allowing_clean DIR "true")
  mv plots/window.pdf plots/{{output_dir}}/window_{{replace(DIR, "/", "_")}}.pdf

# ==================================================

# paper_latent: (latent "paper_main" "94-108_9-25\",\"-8-9_-6-6")
paper_latent: (latent "paper_main" "91-114_13-41") dolphin

# paper_windows DIR="plots/paper_main/91-114_13-41":
#   #!/usr/bin/env bash

#   fd . {{DIR}} --min-depth 1 --max-depth 1 --type d --print0 |
#   while IFS= read -r -d '' d; do
#     # get rid of the first component ("plots") in the path
#     case "$d" in
#       /*) rel="${d#/*/}" ;;  # /a/b/c -> b/c
#       *)  rel="${d#*/}"  ;;  # a/b/c  -> b/c
#     esac
#     echo "Doing stuff in: $rel"
#     just latex::window "$rel"
#     sleep 1
#   done

paper_embedded_windows DIR="plots/paper_main/91-114_13-41":
  #!/usr/bin/env bash

  fd . {{DIR}} --min-depth 1 --max-depth 1 --type d --print0 |
  while IFS= read -r -d '' d; do
    # get rid of the first component ("plots") in the path
    case "$d" in
      /*) rel="${d#/*/}" ;;  # /a/b/c -> b/c
      *)  rel="${d#*/}"  ;;  # a/b/c  -> b/c
    esac
    echo "Doing stuff in: $rel"
    just latex::embedded_window "$rel"
    sleep 1
  done