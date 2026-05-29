# Structural Probes for Spanish Syntax: mBERT & UD AnCora

This repository contains an experimental replication and extension of
**"A Structural Probe for Finding Syntax in Word Representations"**
(Hewitt & Manning, NAACL 2019), adapted to **Spanish** using the
**UD_Spanish-AnCora** treebank and **multilingual BERT** (mBERT) as the
encoder.

## What we investigate

1. Does mBERT encode Spanish dependency syntax in the geometry of its
   contextual word representations, despite never being trained with any
   syntactic supervision?
2. Across which mBERT layers is this syntactic signal strongest?
3. Is the syntactic structure embedded as a **rigid geometric shape** in
   the representation space, or does it require non-uniform scaling
   (anisotropic distortion) to be exposed?

To address (3) we extend the original methodology with an **Isometric
Probe**: a structural probe whose projection matrix is constrained to be
orthogonal (`B^T B = I`). If the syntax tree exists as a "ready-to-use"
shape in the embedding space, this constrained probe should match the
unconstrained linear probe; if instead the probe needs to stretch
specific directions, the orthogonal probe will underperform.

## Dataset

* **UD_Spanish-AnCora**, distributed via Universal Dependencies under
  the **CC BY 4.0** license. Newswire text, originally annotated in the
  AnCora project (Universitat de Barcelona; Taulé et al., 2008),
  converted to dependencies for CoNLL-2009 and then to Universal
  Dependencies.
* GitHub: <https://github.com/UniversalDependencies/UD_Spanish-AnCora>
* Splits used here:

  | split | sentences | conllu lines |
  |-------|----------:|-------------:|
  | train | 14 287    | 529 439      |
  | dev   | 1 654     |  62 444      |
  | test  | 1 721     |  62 841      |

  The exact UD release should be recorded in `data/es_ancora/UD_VERSION`
  (we have not pinned a release yet — please add the version reported in
  AnCora's own README when downloading).

## Encoder

* `bert-base-multilingual-cased` from HuggingFace, frozen.
* 12 transformer blocks + 1 input embedding layer = 13 hidden states
  per token.
* Sub-word aggregation: configurable (`mean` or `first`). Hewitt &
  Manning (2019) use `first`; we use `mean` by default but expose
  `--aggregation first` in `generate_embeddings.py` for direct
  comparability.

## Repository layout

```
scripts/
  conllu_to_text.py        # CoNLL-U -> whitespace-tokenized text, dropping contraction range rows
  generate_embeddings.py   # mBERT contextual embeddings (all 13 layers) -> HDF5
  data.py                  # PyTorch DataLoaders; auto-detects single/multi-layer HDF5
  task.py                  # ParseDistanceTask, ParseDepthTask
  probe.py                 # Linear and Isometric (orthogonal) PSD probes
  loss.py                  # L1 loss for distance and depth
  regimen.py               # train/predict loop with early stopping
  reporter.py              # Spearman, UUAS, MST visualization, tikz dump
  run_experiment.py        # single-seed, single-layer experiment driver
  run_multiseed.py         # repeat one config across N seeds and aggregate
  run_layer_sweep.py       # train one probe per layer (curve over depth)
  calc_condition_number.py # SVD-based diagnostic of probe geometry
data/es_ancora/            # CoNLL-U files and derived .txt / .hdf5
es_ancora.yaml             # default experiment configuration
results/                   # outputs (created at run time)
references.bib             # bib entries for the relevant literature
```

## Methodological choices and known caveats

### Punctuation filtering for UUAS
We follow Hewitt & Manning (2019) in excluding punctuation tokens from the
minimum spanning tree before computing UUAS. We use the **UPOS** tag
`PUNCT` (Universal Dependencies) so the same code works for Spanish,
English, and any other UD treebank.

> Earlier versions of this codebase filtered XPOS using PTB-style English
> tags (`","`, `"."`, etc.), which silently included Spanish punctuation
> in the UUAS computation. UUAS numbers reported before this fix are not
> directly comparable to the literature.

### Sub-word aggregation
mBERT uses WordPiece sub-words. For each whitespace-separated word, the
default behaviour is to **average** the contextual vectors of all its
sub-pieces. The original Hewitt-Manning code instead takes only the
**first** piece. Switch with `--aggregation first` in
`generate_embeddings.py`.

### Contractions and enhanced empty nodes
AnCora UD has two kinds of CoNLL-U rows that should NOT be fed to mBERT
as words:

* **Contraction range rows** (`13-14 al`, `60-61 celebrarlos`). The
  individual sub-word rows (13 `a`, 14 `el`) carry the real surface
  forms; the range row is just metadata.
* **Enhanced empty nodes** (`8.1 _ _ PRON`). These represent
  syntactically projected but surface-absent tokens — typically dropped
  Spanish subjects (`él/ella`). Their FORM is `_`. AnCora has 6499 such
  rows in train, 780 in dev, 833 in test.

Both kinds of rows are filtered with the same predicate (`ID is not a
pure integer`) in `conllu_to_text.py` and again in `data.py` for
defence-in-depth. The text emitted is `... de el Banco Central .`
rather than `... del Banco Central .`, and empty nodes never appear at
all.

> Earlier versions of `conllu_to_text.py` only filtered range rows, so
> empty-node FORMs (literal `_`) were emitted as words and mBERT was
> producing embeddings for them, with the probe then learning syntactic
> geometry over fake tokens. Any number reported before this fix is
> contaminated by ~5% spurious tokens (proportion of empty nodes in
> AnCora) and is not directly comparable to numbers from the corrected
> pipeline.

### Tree distance computation
Tree distances are computed exactly with Floyd-Warshall on the undirected
adjacency matrix of the dependency tree. Disconnected components in the
gold annotation (data quality issues) are masked with a large finite
value and a `warnings.warn` is emitted so they can be inspected.

### Layer choice
mBERT has 13 hidden states (input embeddings + 12 transformer layers).
Hewitt & Manning (2019) report that English BERT's syntactic information
peaks around layer 7–8. We default the YAML to `model_layer: 7` and
recommend using `scripts/run_layer_sweep.py` to plot the full curve when
running new experiments.

## Reproduction

### 1. Install
```
pip install -r requirements.txt
```

### 2. Download UD AnCora
```
git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cp UD_Spanish-AnCora/es_ancora-ud-{train,dev,test}.conllu data/es_ancora/
```

### 3. Pre-process (drop contraction range rows)
```
for split in train dev test; do
  python -m scripts.conllu_to_text \
    data/es_ancora/es_ancora-ud-${split}.conllu \
    data/es_ancora/es_ancora-ud-${split}.txt
done
```

### 4. Generate embeddings (all 13 layers)
```
for split in train dev test; do
  python -m scripts.generate_embeddings \
    data/es_ancora/es_ancora-ud-${split}.txt \
    data/es_ancora/es_ancora-ud-${split}.hdf5 \
    --layers all --aggregation mean
done
```
Add `--random-init` to produce embeddings from a randomly initialised
mBERT (control baseline).

### 5. Train and report
Single seed, single layer (as configured in `es_ancora.yaml`):
```
python -m scripts.run_experiment es_ancora.yaml --seed 0
```
Layer sweep (one probe per layer):
```
python -m scripts.run_layer_sweep es_ancora.yaml --layers 0 1 2 3 4 5 6 7 8 9 10 11 12
```
Multi-seed for a fixed config:
```
python -m scripts.run_multiseed es_ancora.yaml --seeds 0 1 2 3 4
```
Set `probe.isometric: true` in the YAML to switch to the orthogonal probe,
and `reporting.evaluate_test: true` once you are ready to commit to a
final number on the held-out test set.

### 6. Geometric diagnostic
```
python -m scripts.calc_condition_number results/es_ancora/.../predictor.params
```
For the linear probe this reports the condition number κ of the
projection. For the isometric probe κ should be ≈ 1.0 by construction;
the script will warn if it isn't.

## Results

> **Status: pending re-run.** The numbers previously reported in this
> README were computed with (a) layer 12 only, (b) the buggy English
> punctuation filter, and (c) `maximum_rank: 128` (not 768 as previously
> stated), single seed. After the fixes in this codebase the numbers
> need to be regenerated. The runbook in `RUNBOOK.md` lists the
> experiments to run.

When refreshed, the table will report, for each (probe, layer) pair:

* dev / test Spearman ρ averaged over sentence lengths 5–50
* dev / test UUAS (Undirected Unlabeled Attachment Score)
* mean ± std across ≥ 3 seeds
* condition number κ of the projection matrix

A second table will report the same metrics for the **random-init mBERT
baseline** (control: how much does the structural probe learn from
random representations?).

## Limitations and open work

* **No control task** in the strict sense of Hewitt & Liang (2019).
  The random-init baseline gives a representation-level control;
  implementing a word-type-level control task with selectivity would
  strengthen the claims. Tracked but not implemented.
* **Single encoder.** Comparing mBERT to BETO, RoBERTa-bne, and XLM-R
  would test the "shared multilingual capacity hurts Spanish syntax"
  hypothesis seriously; this is left as future work.
* **Only the parse-distance probe is documented end-to-end.** The
  `parse-depth` task is implemented but not the focus of this study.
* **CPU is enough to run a single seed at a single layer in a few
  hours.** A full layer sweep × 5 seeds × 2 probes is ~130 runs; budget
  GPU time for a serious sweep.

## References

See `references.bib` for full BibTeX. Key references:

* Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding
  Syntax in Word Representations. *NAACL*. <https://aclanthology.org/N19-1419/>
* Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes
  with Control Tasks. *EMNLP*. <https://aclanthology.org/D19-1275/>
* Chi, E. A., Hewitt, J., & Manning, C. D. (2020). Finding Universal
  Grammatical Relations in Multilingual BERT. *ACL*.
* Taulé, M., Martí, M. A., & Recasens, M. (2008). AnCora: Multilevel
  Annotated Corpora for Catalan and Spanish. *LREC*.
* UD_Spanish-AnCora: <https://universaldependencies.org/treebanks/es_ancora/index.html>
