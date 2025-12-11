# Structural Probes for Spanish Syntax: mBERT & AnCora

This repository contains an experimental replication of the paper **"A Structural Probe for Finding Syntax in Word Representations"** (Hewitt & Manning, 2019), adapted specifically for the **Spanish language**.

The primary objective is to investigate whether multilingual language models (such as **mBERT**) encode Spanish syntactic structure (dependency trees) within their vector geometry, without having been explicitly trained with syntactic supervision.

Additionally, we extend the original methodology by introducing an **Isometric Probe**. This variant constrains the probe to be a rigid rotation (orthogonal transformation), testing the hypothesis that syntax trees exist as "ready-to-use" geometric shapes in the embedding space, rather than requiring distortion (non-uniform scaling) to be revealed.

## 📊 Results

We trained two types of probes for the Parse Distance task using embeddings from the 12th layer of `bert-base-multilingual-cased`:
1. Linear Probe (Baseline): The standard probe allowing arbitrary linear transformation (scaling + rotation).
2. Isometric Probe (New): A constrained probe allowing only rotation and reflection ($B^T B = I$).

Despite initial alignment challenges regarding Spanish contractions, our results show a strong Spearman correlation, suggesting that mBERT captures significant syntactic information for the Spanish language.

## Experimental Setup
* **Model:** mBERT (`bert-base-multilingual-cased`), frozen weights.
* **Layer:** Last hidden layer (Layer 12).
* **Task:** Parse Distance (predicting syntactic distance between word pairs).
* **Training:** 30 epochs, L1 Loss, Batch size 32.

We evaluated the probe using UUAS (Undirected Unlabeled Attachment Score) and Spearman Correlation.
| Probe Type | Constraint | Spearman $\rho$ | UUAS (accuracy) | Interpretation |
| :--- | :---: | :--- | :--- | :--- |
| **Linear** | **None** | **0.735** | **0.504** | **Latent Structure Found.** Confirms that mBERT captures the geometric distance between syntactically related words in Spanish. The probe successfully recovers syntax by re-weighting dimensions. Outperforms the random baseline (0.44), demonstrating latent structural learning.|
| **Isometric** | **Orthogonal (Orthogonal ($B^TB=I$))** | **0.658** | **0.375** | **Geometric Mismatch.** The significant drop (-25%) indicates the raw embedding geometry is too distorted to represent trees directly.|

## Geometric Stability Analysis

The performance gap between the Linear and Isometric probes suggests that the embedding space suffers from anisotropy (the "representation cone" problem).

We analyzed the transformation matrix of the successful Linear Probe and found a Condition Number ($\kappa$) of 12.30.
* **Implication:** mBERT suppresses Spanish syntactic dimensions by a factor of ~12 relative to dominant semantic/frequency dimensions.
* **Conclusion:** Spanish syntax is present in mBERT, but it is not "geometric" in the Euclidean sense. It exists as extractable features that must be amplified (stretched) by the probe to be used.

> **Note:** The discrepancy in UUAS compared to the original English paper is attributed to mBERT sharing capacity across 104 languages and the richer morphology of Spanish.


## 🛠️ Modifications & Engineering

To adapt the original 2019 codebase to a modern environment and the specific linguistic features of Spanish, the following key implementations were made:

### 1. Data Alignment (Critical Fix)
Spanish contains contractions (e.g., *del* = *de* + *el*, *al* = *a* + *el*) that are split in the Universal Dependencies (CoNLL-U) format using range indices (e.g., `1-2`). This caused a severe misalignment between the tokenized BERT embeddings and the gold-standard labels. 
* **Solution:** We implemented a filtering strategy in `conllu_to_text.py` and `data.py` to strictly ignore range indices, ensuring a 1-to-1 mapping between the embedding subwords and the syntactic labels.

### 2. Mathematical Optimization
The original distance calculation algorithm in `task.py` was iterative and prone to infinite loops when encountering cyclic dependencies in the annotation data.
* **Solution:** The algorithm was rewritten using a **vectorized Floyd-Warshall algorithm** with NumPy. This ensures robustness against cycles and reduced the pre-computation time from minutes to seconds.

### 3. Library Migration
* **Transformers:** Replaced the obsolete `pytorch-pretrained-bert` with the modern HuggingFace `transformers` library.
* **Windows/UTF-8:** Enforced `utf-8` encoding across all file I/O operations to correctly handle Spanish accents and special characters on Windows systems.


## 🚀 Reproduction Steps
### 1. Requirements
Install the necessary Python packages:
```
Bash

pip -r install requirements.txt
```
### 2. Dataset Download
Download the Universal Dependencies Spanish AnCora corpus (v2.x) from the official repository: UD_Spanish-AnCora GitHub.

Place the .conllu files in the data/es_ancora/ folder:

es_ancora-ud-train.conllu

es_ancora-ud-dev.conllu

es_ancora-ud-test.conllu

### 3. Data Cleaning & Alignment
Run the custom cleaning script to extract raw text and remove contraction ranges (fixing the alignment issue):
```
Bash

python -m scripts.conllu_to_text data/es_ancora/es_ancora-ud-train.conllu data/es_ancora/es_ancora-ud-train.txt
python -m scripts.conllu_to_text data/es_ancora/es_ancora-ud-dev.conllu data/es_ancora/es_ancora-ud-dev.txt
python -m scripts.conllu_to_text data/es_ancora/es_ancora-ud-test.conllu data/es_ancora/es_ancora-ud-test.txt
```
### 4. Embedding Generation
Pre-compute the mBERT embeddings. This freezes the model layers into HDF5 files for faster training.
```
Bash

python -m scripts.generate_embeddings data/es_ancora/es_ancora-ud-train.txt data/es_ancora/es_ancora-ud-train.hdf5
python -m scripts.generate_embeddings data/es_ancora/es_ancora-ud-dev.txt data/es_ancora/es_ancora-ud-dev.hdf5
python -m scripts.generate_embeddings data/es_ancora/es_ancora-ud-test.txt data/es_ancora/es_ancora-ud-test.hdf5
```
### 5. Running the Experiment
Train the structural probe using the configuration file:
```
Bash

python -m scripts/run_experiment es_ancora.yaml
```

**To run the Isometric Probe:** Update your `es_ancora.yaml` configuration file to include the `isometric` flag. This will trigger the hard orthogonality constraint on the probe:

```yaml
probe:
  task_signature: word_pair
  task_name: parse-distance
  maximum_rank: 768
  psd_parameters: True
  diagonal: False
  params_path: predictor.params
  isometric: True  # <--- Set to True for Isometric Probe
```

## 📈 Visualization
Upon completion, results are saved in example/results/es_ancora. You will find:

dev.spearmanr: Quantitative correlation metrics.

*.png (Heatmaps): Visual comparison of distance matrices (Gold vs. Predicted).

*.tikz: LaTeX code to generate vector graphics of the reconstructed syntactic trees.

## 📄 References
If you use this code or methodology, please cite the original paper and the dataset:


@inproceedings{hewitt2019structural,
  title={A Structural Probe for Finding Syntax in Word Representations},
  author={Hewitt, John and Manning, Christopher D},
  booktitle={North American Chapter of the Association for Computational Linguistics (NAACL)},
  year={2019}
}

@inproceedings{taule2008ancora,
  title={AnCora: Multilevel Annotated Corpora for Catalan and Spanish},
  author={Taul{\'e}, Mariona and Mart{\'i}, Maria Ant{\`o}nia and Recasens, Marta},
  booktitle={LREC},
  year={2008}
}




















