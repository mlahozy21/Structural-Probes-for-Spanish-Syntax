# Instructions for Claude Code

This project is a structural-probe study (Hewitt & Manning, 2019, NAACL) on
mBERT for Spanish syntax, using UD_Spanish-AnCora. It is being prepared
for a PhD interview demo. The user's environment is **Windows 11 / Python
3.13 / CPU only / PowerShell**. The user wants the full experimental
pipeline executed end-to-end.

Read `README.md` for the project description and `RUNBOOK.md` for the
intended experiment sequence and cost estimates. This file gives you the
operational instructions for actually running everything.

---

## Operating constraints to respect

- **Shell:** PowerShell (not bash). Loops use `foreach ($s in 'a','b') { ... }`,
  not `for s in a b; do ... done`.
- **Python:** `python` should refer to the venv in `.\.venv\` once created.
  Always activate before running anything: `.\.venv\Scripts\Activate.ps1`.
- **Long-running steps:** `generate_embeddings.py` for train can take 30-60
  minutes on CPU; the layer sweep can take several hours. Do not abort
  unless the user asks. Surface progress via tqdm output. If you must
  break a long step into chunks (e.g. for tooling timeouts) do so per
  split (`train`, `dev`, `test` separately) — do NOT subdivide a single
  split, because partial HDF5 files break alignment.
- **Disk space:** train.hdf5 with `--layers all` is ~5-6 GB. Verify free
  space (>= 20 GB) before launching tanda 2.
- **OneDrive sync:** the project lives in OneDrive. If you write files via
  scripts and they appear truncated to other tools, run `sync` (Linux) or
  wait a few seconds before re-reading. On Windows this is usually not an
  issue.

---

## Step 0 — Environment setup

```powershell
# From the project folder
.\setup.ps1
```

If `setup.ps1` errors on execution policy, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If the user does not have `setup.ps1`, do the equivalent manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python -c "import torch, transformers, h5py, scipy, yaml, numpy; print('OK')"
```

**Stop criterion:** `python -c "..."` prints `OK`. If imports fail,
inspect the error and fix before proceeding.

---

## Step 1 — Pre-process CoNLL-U into whitespace-tokenized text

```powershell
foreach ($s in 'train','dev','test') {
    python -m scripts.conllu_to_text data/es_ancora/es_ancora-ud-$s.conllu data/es_ancora/es_ancora-ud-$s.txt
}
```

**Stop criterion:** the script prints `Wrote N sentences. Skipped M
non-integer-ID rows ...` for each split. Expect roughly:
- train: 14287 sentences, ~16628 skipped (10129 contractions + 6499 empty nodes)
- dev:   1654 sentences, ~2042 skipped
- test:  1721 sentences, ~2095 skipped

---

## Step 2 — Generate mBERT embeddings (all 13 layers)

This is the longest step. The first call also downloads mBERT (~700 MB
from HuggingFace) into the user's HuggingFace cache. Subsequent calls
reuse the cache.

```powershell
foreach ($s in 'train','dev','test') {
    python -m scripts.generate_embeddings `
        data/es_ancora/es_ancora-ud-$s.txt `
        data/es_ancora/es_ancora-ud-$s.hdf5 `
        --layers all --aggregation mean --batch-size 8
}
```

**Time estimate (CPU):** ~5 min download (one-off) + ~5-10 min for dev/test
each + ~30-50 min for train. Total: ~45-90 min.

**Stop criterion:** three `Wrote ...hdf5.` messages. Sanity-check sizes:

```powershell
Get-ChildItem data\es_ancora\*.hdf5 | Format-Table Name, @{Label="MB";Expression={[math]::Round($_.Length/1MB,1)}}
```

Train should be ~3-6 GB, dev/test ~300-700 MB each.

If you get `Batch contains a sentence longer than the model max length`,
reduce `--batch-size` to 1 to isolate the problem sentence and report it
to the user.

---

## Step 3 — Layer sweep (linear probe across all 13 layers)

```powershell
python -m scripts.run_layer_sweep es_ancora.yaml --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 --seed 0
```

**Time estimate:** ~3-5 hours on CPU. Each layer is a full 30-epoch probe
training with early stopping (patience=4).

**Stop criterion:** the script prints a summary table of all 13 layers
and writes `results/es_ancora/layersweep-<timestamp>/sweep.csv`. Open
the CSV and identify the layer with the highest `dev_spearman_5_50` —
this is the "best layer" used in step 4.

---

## Step 4 — Linear vs. isometric probe at the best layer

After identifying the best layer (call it `L_best`):

```powershell
# Set the best layer in the YAML
(Get-Content es_ancora.yaml) -replace 'model_layer: \d+', "model_layer: $L_best" | Set-Content es_ancora.yaml

# Linear probe
python -m scripts.run_experiment es_ancora.yaml --seed 0
```

Then edit `es_ancora.yaml` to set `isometric: true` (or do it
programmatically):

```powershell
(Get-Content es_ancora.yaml) -replace 'isometric: false', 'isometric: true' | Set-Content es_ancora.yaml
python -m scripts.run_experiment es_ancora.yaml --seed 0
```

After this, restore the YAML to `isometric: false` to leave it in its
default state.

**Time estimate:** ~15-30 min per probe. Total ~30-60 min.

**Stop criterion:** two `predictor.params` files exist, one in
`results/es_ancora/BERT-disk-parse-distance-<ts>/` (linear) and one in
`results/es_ancora/iso/BERT-disk-parse-distance-<ts>/` (isometric).

---

## Step 5 — Geometric diagnostic (condition number)

For the linear probe:

```powershell
$linDir = (Get-ChildItem -Directory results\es_ancora\BERT-disk-parse-distance-* | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
python -m scripts.calc_condition_number "$linDir\predictor.params"
```

For the isometric probe (must pass `--config` so the parametrization can
be reconstructed):

```powershell
$isoDir = (Get-ChildItem -Directory results\es_ancora\iso\BERT-disk-parse-distance-* | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
python -m scripts.calc_condition_number "$isoDir\predictor.params" --config es_ancora.yaml
```

**Stop criterion:** the linear path prints a finite kappa (typically
2-15 depending on layer); the isometric path prints
`OK: orthogonality constraint satisfied` with kappa ~= 1.0000.

---

## Step 6 — Report back to the user

Print a concise summary in chat:

1. The best layer from the sweep (with its dev Spearman and UUAS).
2. The linear vs. isometric comparison at that layer (dev metrics for
   both, with the gap as a percentage).
3. Both condition numbers.
4. The four file paths the user can open: `sweep.csv`, the two
   `predictor.params` and the corresponding metric files.

Do NOT modify `README.md` to insert numbers — let the user decide what
to publish.

---

## Things you should NOT do

- Do not modify `scripts/probe.py` or `scripts/task.py` — they have been
  audited extensively.
- Do not change the YAML's `observation_fieldnames` order — it must
  match CoNLL-U columns 1-10 plus `embeddings`.
- Do not skip Step 1 even if `data/es_ancora/*.txt` already exists —
  earlier versions of the .txt were generated with a buggy filter that
  included empty-node `_` tokens. Always regenerate.
- Do not enable `reporting.evaluate_test: true` in the YAML during the
  layer sweep or initial linear/isometric runs — touch the test set
  ONLY at the very end, when reporting final headline numbers.
- Do not commit changes to git automatically. The user will review and
  commit themselves.

---

## If something fails

| Symptom | Diagnosis | Fix |
|---|---|---|
| `Alignment mismatch: N obs vs M emb` | The `.txt` and `.hdf5` were not regenerated together | Re-run step 1 then step 2 for that split |
| `model_layer=X was not saved in <file>` | YAML asks for a layer not in the HDF5 | Either regenerate with `--layers all`, or change `model_layer` to one that was saved |
| `Sentence longer than model max length` | A sentence > 512 sub-tokens | Reduce `--batch-size 1` to find it; report it to the user — do not auto-truncate |
| OOM during training | Batch too large for CPU RAM | Lower `dataset.batch_size` from 32 to 16 in the YAML |
| `WARNING: legacy 2D HDF5` | Old HDF5 file from before the multi-layer rewrite | Regenerate with `scripts.generate_embeddings ... --layers all` |
| Probe loss does not decrease | Bad embeddings or wrong layer | Verify HDF5 attrs (`h5py` shows `layer_indices` and `model_name`); confirm the right layer is in `model_layer` |
