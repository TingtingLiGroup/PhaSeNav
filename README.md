# Programming Cellular Navigation via Reader-Writer Engineering of Condensate-Targeting Signals

This project provides a complete protein analysis pipeline for ESM representation extraction, SAE sparse encoding, dynamic feature selection, protein classification, and novel constitutive protein prediction\.

## Overview

The entire pipeline is standardized into **four progressive stages**, covering feature construction, model training, standalone evaluation and novel protein prediction:

1. **encode**
Extract ESM residue\-level representations from raw protein sequences and convert dense embeddings into SAE sparse protein features\.

2. **train\+test**
Train dynamic feature selection classification models and complete full training and testing evaluation\.

3. **test**
Standalone inference and evaluation based on saved features and trained model checkpoints\.

4. **new constitute prediction**
Score and rank unseen novel constitutive proteins with pre\-trained models\.

---

## Project Structure

Strictly consistent with the latest project file specifications:

```bash
project/
├── run.sh                  # Pipeline running script
├── config.sh               # Global parameter configuration
├── run_pipeline.py         # Pipeline scheduling entry
├── Sae_classify.py         # Main classification & prediction entry
├── train_classify.py       # Model training & evaluation logic
├── data_prepare.py         # Raw sequence data preprocessing
├── dictionary.py           # Encoding & dictionary utility
├── model_mlp.py            # MLP classification model definition
├── get_fasta.py            # FASTA sequence file processing
├── get_features.py         # ESM & SAE feature extraction
├── environment.yml         # Conda environment configuration
├── dataset/                # Raw & processed dataset storage
├── generation/             # Intermediate files, features, model outputs & results
├── .gitignore
└── README.md
```

---

## Environment \& Dependencies

### Requirements

- Python \>= 3\.9

- PyTorch

- ESM

- NumPy

- Pandas

- Scikit\-learn

- Matplotlib

- tqdm

### Installation

Option 1: Conda environment \(recommended\)

```bash
conda env create -f environment.yml
```

Option 2: Pip installation

```bash
pip install -r requirements.txt
```

---

## Configuration

Modify `config.sh` to customize global experimental parameters:

- CUDA device ID

- Input / output file paths

- Training / inference batch size

- ESM / SAE checkpoint paths

- Feature dimension settings

- Train / test / prediction hyperparameters

Example configuration:

```bash
DEVICE="cuda:1"
FEATURE_DIM=40960
```

---

## Usage

Grant execution permission first:

```bash
chmod +x run.sh config.sh
```

Run the four pipeline stages in sequence:

```bash
# Stage 1: Encode (ESM extraction + SAE sparse feature generation)
./run.sh encode

# Stage 2: Train + Test (Feature selection + model training + evaluation)
./run.sh train_test

# Stage 3: Standalone Test (Evaluate with saved model & features)
./run.sh test

# Stage 4: New Constitute Prediction (Novel protein scoring & ranking)
./run.sh predict
```

---

## Stage Details

### 1\. encode

Integrates original sequence representation extraction and SAE sparse encoding into a single stage\.

**Functions:**

- Load raw protein sequence data

- Extract ESM residue\-level dense representations

- Save train / test tensor files

- Load pre\-trained SAE checkpoint

- Convert dense embeddings to sparse SAE protein features

*Corresponds to original Step 1 \+ Step 2*

### 2\. train\+test

Core stage for dynamic feature selection and classification model development\.

**Functions:**

- Load preprocessed train/test tensors and sparse SAE features

- Screen and select high\-information protein features

- Train one\-vs\-one MLP classifiers

- Evaluate classification performance on test set

- Save optimal feature subsets and model performance records

*Corresponds to original Step 3*

### 3\. test

Independent offline evaluation stage for completed models and features\.

**Functions:**

- Load saved optimal feature files

- Load trained model checkpoints

- Perform test set inference and metric calculation

- Output independent test evaluation results

*Corresponds to original Step 4*

### 4\. new constitute prediction

Prediction stage for unseen novel constitutive proteins without label data\.

**Functions:**

- Load sparse features of new target proteins

- Load screened optimal features and trained models

- Generate prediction scores for each protein

- Output score tables for protein ranking and downstream analysis

*Corresponds to original Step 5*

---

## Output Files

All results are saved in the `generation/` directory:

- `train_tensors.pt` / `test_tensors.pt` — Processed feature tensors

- SAE sparse protein feature `*.pt` files

- `selected_features_*.csv` — Screened optimal feature list

- `best_model_performance_*.csv` — Model evaluation metrics

- `final_protein_scores.csv` — Novel protein prediction scores

- `final_protein_scores_with_uniprot.csv` — Annotated scoring results

---

## Notes

- **encode** unifies ESM representation extraction and SAE sparse encoding to simplify pipeline deployment\.

- **train\+test** is the core experimental stage for model training and feature optimization\.

- **test** is only used for standalone verification of mature models\.

- **new constitute prediction** is specially designed for unscored novel constitutive protein mining\.

---

## License

Add your license information here\.

## Contact

Add your contact information here\.

> (Note: May contain AI-generated content.)
