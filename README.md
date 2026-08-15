# Project README

## 📁 Project Structure

The overall file structure of this project is as follows:

```bash
.
├── dataset/            # Dataset storage directory
├── generation/         # Intermediate files and model output results
├── .gitignore          # Git ignore configuration
├── README.md           # Project documentation
├── Sae_classify.py     # Main classification pipeline script (entry)
├── data_prepare.py     # Data preprocessing script
├── dictionary.py       # Dictionary & encoding tool functions
├── environment.yml     # Conda environment configuration file
├── get_fasta.py        # FASTA sequence file generation/extraction
├── get_features.py     # Data feature extraction script
├── model_mlp.py        # MLP classification model definition
└── train_classify.py   # Model training & experimental script
```

## 🚀 Quick Start

### 1\. Environment Configuration

We provide a `environment.yml` file for one\-click Conda environment creation\. Please execute the following command to build the running environment:

```bash
conda env create -f environment.yml
```

Conda will automatically create the corresponding environment according to the configuration file\.

### 2\. Standard Workflow

The complete running process of the project is divided into **data preprocessing** and **model classification inference**:

#### Step 1: Data Preprocessing

```bash
python data_prepare.py
```

#### Step 2: Run Classification Task \(Main Entry\)

```bash
python Sae_classify.py
```

## 📄 File Function Description

|File / Folder|Function Description|
|---|---|
|`Sae_classify.py`|The main entry script of the project, responsible for running the complete classification pipeline|
|`data_prepare.py`|Realize data cleaning, screening, format conversion and other preprocessing operations|
|`dictionary.py`|Provide general tool functions such as data dictionary construction and encoding conversion|
|`get_fasta.py`|Extract or generate biological sequence files in FASTA format|
|`get_features.py`|Extract effective feature information from original input data for model training|
|`model_mlp.py`|Define the network structure and related parameters of the MLP classification model|
|`train_classify.py`|Responsible for model training, parameter optimization and classification experiment verification|
|`dataset/`|Store original data and processed dataset files required for the experiment|
|`generation/`|Save intermediate generated files, model prediction results and output files|

## 💡 Notes

- **Core Script**: `Sae_classify.py` is the only entry for the classification task, and other scripts are auxiliary tool modules\.

- **Path Configuration**: Before running the code, please check and modify the file input/output paths in the script according to your local environment to avoid path errors\.

- **Environment Dependence**: Please strictly install the environment according to `environment.yml` to ensure the consistency of dependency versions\.

## 📮 Contact \& Feedback

If you have any questions, bugs or optimization suggestions during use, please submit an **Issue** in this repository\.

