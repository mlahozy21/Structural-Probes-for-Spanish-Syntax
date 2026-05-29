# Technical report

`report.tex` is a self-contained LaTeX document that summarises the
experiments described in the project's main `README.md`. It is written
in plain `article` class so it compiles with any standard TeX
distribution (TeX Live, MiKTeX, Overleaf), and uses only widely
available packages (`pgfplots`, `natbib`, `booktabs`, `microtype`).

## Compile

```
cd report
pdflatex report.tex
bibtex   report           # if you switch to a separate .bib file
pdflatex report.tex
pdflatex report.tex
```

The bibliography is currently inlined as a `thebibliography`
environment, so a single `pdflatex` pass is enough. To use the project's
`references.bib` instead, replace the `\begin{thebibliography}...\end{thebibliography}`
block at the bottom of `report.tex` with `\bibliography{../references}`
and run the full `pdflatex / bibtex / pdflatex / pdflatex` cycle.

## Switch to a conference template

The current document uses `\documentclass{article}` for portability.
To switch to ACL / EMNLP, NeurIPS or IEEE style:

* **ACL/EMNLP** — drop the official `acl.sty` next to `report.tex`,
  replace the document class line with
  `\documentclass[11pt,a4paper]{article}` followed by
  `\usepackage[review]{acl}`, and remove `\usepackage{geometry}`.
* **NeurIPS** — replace with `\documentclass{article}` + the
  `neurips_2024.sty` from the official NeurIPS template.
* **IEEE conference** — replace with `\documentclass[conference]{IEEEtran}`
  and adapt the bibliography style to `\bibliographystyle{IEEEtran}`.

The body of the report should compile under any of these classes
without modification beyond the figure size.

## Source numbers

All numerical values in the report come from the layer sweep run on
2026-05-09 (UTC) on a Google Colab NVIDIA RTX PRO 6000 Blackwell
(96 GB VRAM):

* Layer sweep CSV: `results/es_ancora/layersweep-20260509-182723/sweep.csv`
* Linear probe at layer 7: `results/es_ancora/BERT-disk-parse-distance-2026-5-9-19-4-58-5282/`
* Isometric probe at layer 7: `results/es_ancora/iso/BERT-disk-parse-distance-2026-5-9-19-7-55-820170/`

If you re-run the experiments and want the report to refresh
automatically, replace the hard-coded numbers in the `results` section
with `\input` of small `.tex` snippets generated from the CSV.
