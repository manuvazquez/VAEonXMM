output_dir := "figures"

latexmk_command := "latexmk -lualatex -quiet -gg"
latexmk_outdir := "-outdir=" + output_dir

default:
  just -f {{source_file()}} --list

all: (reference_vs_non-reference "paper_main") (reference_vs_non-reference "ae_z_10") (embedded_window "paper_main/106-117_-16-2/11099")

[private]
reference_vs_non-reference_with_clean DIR CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\dataFile}{"{{DIR}}"/tSNE.csv}" -jobname=reference_vs_non-reference_{{DIR}} {{latexmk_outdir}} -cd "plots/reference_vs_non-reference.tex"

reference_vs_non-reference DIR: (reference_vs_non-reference_with_clean DIR) (reference_vs_non-reference_with_clean DIR "true")

# reference_vs_non-reference DIR:
#     {{latexmk_command}} "-usepretex=\\newcommand{\\dataFile}{"{{DIR}}"/tSNE.csv}" -jobname=reference_vs_non-reference_{{DIR}} {{latexmk_outdir}} -cd "plots/reference_vs_non-reference.tex"
#     # cleaning auxiliary files
#     {{latexmk_command}} "-c" -jobname=reference_vs_non-reference_{{DIR}} {{latexmk_outdir}} -cd "plots/reference_vs_non-reference.tex"

# reference_vs_non-reference DIR:

[private]
embedded_window_with_clean DIR CLEAN="false":
    {{latexmk_command}} {{if CLEAN == "true" {"-c"} else {""} }} "-usepretex=\\newcommand{\\subdir}{"{{DIR}}"}" -cd "plots/embedded_window.tex"

embedded_window DIR: (embedded_window_with_clean DIR) (embedded_window_with_clean DIR "true")
  mv plots/embedded_window.pdf plots/{{output_dir}}/embedded_window_{{replace(DIR, "/", "_")}}.pdf

# paper_main/106-117_-16-2/11099/