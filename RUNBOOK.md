# Runbook: refreshing all numbers after the code overhaul

This document is the playbook for re-running the experiments now that the
following bugs/limitations have been fixed in the code:

* **CRITICAL — enhanced empty nodes were being fed to mBERT as `_`
  tokens.** AnCora's UD release contains 6499 enhanced empty nodes in
  train, 780 in dev, 833 in test (e.g. dropped Spanish subjects like
  pro-`él`). Their FORM column is `_`. The previous `conllu_to_text.py`
  emitted these as the literal `_` word, so mBERT received `_`
  embeddings at those positions, and the trained probe was trying to
  learn syntactic geometry over **fake tokens**. Both
  `conllu_to_text.py` and `data.py` now skip every CoNLL-U row whose ID
  is not a pure integer (range contractions AND empty nodes). All .txt
  and .hdf5 files MUST be regenerated; otherwise alignment errors will
  raise loudly at training time (the new alignment assertion in
  `data.py` will trigger on the old .txt files);
* punctuation filtering for UUAS now uses UD's universal `PUNCT` tag (the
  previous code silently included Spanish punctuation, since it filtered
  PTB-style English XPOS tags);
* `generate_embeddings.py` now stores **all 13 mBERT layers** rather than
  only the last one, so layer-wise probing is possible;
* `data.py` correctly indexes the requested layer from a 3D HDF5 (and
  remains backward-compatible with the old 2D dumps);
* test-set evaluation is now a YAML option (`reporting.evaluate_test:
  true`) instead of commented-out code;
* `run_multiseed.py` and `run_layer_sweep.py` provide proper harnesses
  for the two analyses you actually want to report;
* `calc_condition_number.py` works for both the linear and the
  isometric probe;
* `task.py` now warns instead of silently masking annotation errors;
* `run_experiment.py` no longer skips seeding when `--seed 0` (the old
  `if cli_args.seed:` check was a Python-falsy gotcha, so seed=0 runs
  were not actually reproducible).

## Cost estimates (CPU only)

Rough orders of magnitude on a modern laptop CPU. Multiply by ~5–10x if
the machine is older or congested. Divide by ~10x with a single GPU.

| step                                           | wall-clock |
|------------------------------------------------|-----------:|
| `conllu_to_text` for all 3 splits              | seconds    |
| `generate_embeddings --layers all` for all 3   | 3–6 h      |
| training one probe (1 layer, 1 seed, 30 epochs)| 30–90 min  |
| layer sweep (13 layers, 1 seed)                | 6–18 h     |
| 5 seeds at one layer                           | 2.5–7.5 h  |

Scope your time accordingly. **For the PhD interview**, a single layer
sweep (linear probe, 1 seed) and the existing isometric-vs-linear
comparison (now with correct UUAS) are the highest-value outputs.

## Step-by-step

### 0. Sanity check the code

```
python -c "from scripts import probe, task, reporter, data; print('ok')"
```

If this fails, fix imports before touching experiments.

### 1. Re-pre-process the corpus

The CoNLL-U files are unchanged, but re-running guarantees a clean text
split.

```
for split in train dev test; do
  python -m scripts.conllu_to_text \
    data/es_ancora/es_ancora-ud-${split}.conllu \
    data/es_ancora/es_ancora-ud-${split}.txt
done
```

### 2. Regenerate embeddings (multi-layer)

```
for split in train dev test; do
  python -m scripts.generate_embeddings \
    data/es_ancora/es_ancora-ud-${split}.txt \
    data/es_ancora/es_ancora-ud-${split}.hdf5 \
    --layers all --aggregation mean --batch-size 8
done
```

The `.hdf5` files will grow ~13x (all layers). If disk is tight, request
just a few layers, e.g. `--layers 0,4,7,8,12`.

### 3. Layer sweep with the linear probe

This produces the canonical "Spearman vs. layer" curve.

```
python -m scripts.run_layer_sweep es_ancora.yaml \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 --seed 0
```

Output: `results/es_ancora/layersweep-<timestamp>/sweep.csv` and one
subdirectory per layer with the usual `dev.spearmanr`, `dev.uuas`,
heatmaps, etc.

> If 13 runs is too much CPU, restrict to `--layers 0 4 7 8 12`. That
> still produces the *shape* of the curve.

### 4. Pick the best layer and lock the YAML

Inspect `sweep.csv`. Set `model.model_layer` in `es_ancora.yaml` to the
layer with the highest dev Spearman (expected: 7 or 8).

### 5. Multi-seed comparison: linear vs isometric

This produces the headline finding (does the orthogonality constraint
hurt? by how much, with std bars?).

```
# Linear probe
python -m scripts.run_multiseed es_ancora.yaml --seeds 0 1 2 3 4

# Edit es_ancora.yaml: probe.isometric: true
python -m scripts.run_multiseed es_ancora.yaml --seeds 0 1 2 3 4
```

Both write `aggregate.json` with mean ± std for dev_uuas and
dev_spearman.

### 6. Random-init baseline

The single most important control: how much of the structural probe's
performance comes from learning to project random vectors?

```
for split in train dev test; do
  python -m scripts.generate_embeddings \
    data/es_ancora/es_ancora-ud-${split}.txt \
    data/es_ancora/es_ancora-ud-${split}.random.hdf5 \
    --layers all --random-init
done
```

Then point the YAML's `dataset.embeddings.{train,dev,test}_path` at the
`.random.hdf5` files (or duplicate the YAML), pick the same layer, and
run `run_multiseed` again. Expectation: substantially worse than the
real mBERT, but not zero — the probe alone has some capacity.

### 7. Geometric diagnostic

For each saved probe, compute the condition number:

```
python -m scripts.calc_condition_number \
    results/es_ancora/multiseed-.../seed0/predictor.params
```

For the isometric probe the script will print `OK: orthogonality
constraint satisfied` (κ ≈ 1.0).
For the linear probe, the original 12.30 figure should be re-measured
with the corrected setup (correct layer, correct rank, correct
punctuation filtering).

### 8. Touch the test set ONCE, at the very end

When you have committed to a single best (layer, probe, rank) and
multi-seed numbers on dev, set `reporting.evaluate_test: true` in the
YAML and re-run the *exact* configuration used to pick the model. Do
this **once**. Report test numbers in the README and in the slides.

## What to report in the slides

From the experiments above, the strongest 8-minute story is:

1. **Layer-wise Spearman curve** — visual, immediate, and a textbook
   probing-study figure. Annotate the peak.
2. **Linear vs Isometric at the best layer**, table with mean ± std over
   5 seeds, both for dev and test. Highlight the gap as evidence for the
   "syntax is suppressed, not isometric" claim.
3. **Condition number** of the linear probe at the best layer. One
   sentence interpretation: "the probe needs to amplify some directions
   κ× more than others to expose syntax".
4. **Random-init baseline** as the floor. One bar in the comparison plot.

That's enough material for 8 minutes without rushing, and it lines up
exactly with Eje 1 of the Orange PhD offer (probing, ablation,
interpretability).
