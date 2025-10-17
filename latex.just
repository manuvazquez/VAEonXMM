output_dir := "figures"

latexmk_command := "latexmk -lualatex -quiet -gg"
latexmk_outdir := "-outdir=" + output_dir

default:
  just -f {{source_file()}} --list

# all: (reference_vs_non-reference "paper_main") (reference_vs_non-reference "ae_z_10") (embedded_window "paper_main/106-117_-16-2/11099")

all: (reference_vs_non-reference "paper_main" "44-58_91-100") (reference_vs_non-reference "ae_z_10") (embedded_window "paper_main/106-117_-16-2/11099") (validation "paper_main") (roc "paper_main" "vae") (roc "paper_main" "ar")

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
validation_allowing_clean DIR CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\subdir}{"{{DIR}}"}" -cd "plots/validation.tex"

validation DIR: (validation_allowing_clean DIR) (validation_allowing_clean DIR "true")
  mv plots/validation.pdf plots/{{output_dir}}/validation_{{replace(DIR, "/", "_")}}.pdf

# ==================================================

[private]
roc_allowing_clean DIR ALGORITHM CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataDir}{"{{DIR}}"}\\newcommand{\\algo}{"{{ALGORITHM}}"}" -cd "plots/roc.tex"

roc DIR ALGORITHM: (roc_allowing_clean DIR ALGORITHM) (roc_allowing_clean DIR ALGORITHM "true")
  mv plots/roc.pdf plots/{{output_dir}}/{{ALGORITHM}}_roc_{{replace(DIR, "/", "_")}}.pdf

# roc DIR ALGORITHM: (roc_allowing_clean DIR ALGORITHM)