input_dir := join(justfile_directory(),"plots")
output_dir := join(justfile_directory(),"figures")

latexmk_command := "latexmk -lualatex -quiet -gg"
latexmk_outdir := "-outdir=" + output_dir

# foo:
#   ls {{justfile_directory()}}

default:
  just -f {{source_file()}} --list

all: \
  (reference_vs_non-reference "paper_main") (reference_vs_non-reference "ae_z_10") \
  (roc "paper_main" "vae") (roc "paper_main" "ar") \
  paper_embedded_windows \
  paper_latent
  # cp {{input_dir}}/VAE.tex {{input_dir}}/{{output_dir}}
  ln -sf "$(readlink -f {{input_dir}}/VAE.tex)" {{input_dir}}/{{output_dir}}/VAE.tex
  ln -sf "$(readlink -f {{input_dir}}/AR.tex)" {{input_dir}}/{{output_dir}}/AR.tex

dolphin:
  dolphin plots/{{output_dir}} &

# ==================================================

[private]
reference_vs_non-reference_allowing_clean DIR ZOOM_SUBDIR="" CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataDir}{"{{DIR}}"}"{{if ZOOM_SUBDIR!="" {"\\\\newcommand{\\\\zoomSubdir}{"+ZOOM_SUBDIR+"}"} else {""} }} -jobname=reference_vs_non-reference_{{DIR}} {{latexmk_outdir}} -cd "plots/reference_vs_non-reference.tex"

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

east_region := "east"
north_region := "north"
center_region := "center"
south_region := "south"
west_region := "west"

paper_latent: (latent "paper_main" "\"" +
        east_region + 
        "," + 
        north_region +  
        "," + 
        center_region + 
        "," + 
        west_region +         
        "," + 
        south_region + 
        "\"") 
        dolphin

embedded_windows_dir DIR="plots/paper_main/91-114_13-41":
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

paper_embedded_windows: \
  (embedded_windows_dir "plots/paper_main/" + east_region)\
  (embedded_windows_dir "plots/paper_main/" + north_region)\
  (embedded_windows_dir "plots/paper_main/" + center_region)\
  (embedded_windows_dir "plots/paper_main/" + west_region)\
  (embedded_windows_dir "plots/paper_main/" + south_region)\
  dolphin
