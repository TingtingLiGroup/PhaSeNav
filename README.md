.
├── dataset/                # Dataset files
├── generation/             # Generated intermediate files or outputs
├── .gitignore
├── README.md
├── Sae_classify.py         # Main script for classification
├── data_prepare.py         # Data preprocessing
├── dictionary.py           # Dictionary / encoding utilities
├── environment.yml         # Conda environment configuration
├── get_fasta.py            # FASTA file generation or extraction
├── get_features.py         # Feature extraction
├── model_mlp.py            # MLP model definition
└── train_classify.py       # Training script for classification

Main Script
The primary script in this repository is:

bash python Sae_classify.py

This is the main entry point for running the classification task.

File Description
Sae_classify.py
Main script for the classification pipeline.
data_prepare.py
Used for data preprocessing and preparation.
dictionary.py
Contains dictionary or encoding-related utility functions.
get_fasta.py
Used to extract or generate FASTA-format sequence files.
get_features.py
Used to extract features from input data.
model_mlp.py
Defines the MLP model used in the classification task.
train_classify.py
Used for model training and classification-related experiments.
dataset/
Contains dataset files.
generation/
Contains generated files, intermediate outputs, or results.
Environment Setup
It is recommended to create the Conda environment using the provided environment.yml file:

bash conda env create -f environment.yml
If the environment name is already specified in environment.yml, Conda will create it automatically.

Basic Workflow
A typical workflow for this project is:

Prepare the data
bash python data_prepare.py

Run the main classification script
bash python Sae_classify.py


Notes
The main script for this project is Sae_classify.py.
Other scripts are supporting scripts for preprocessing, feature extraction, model training, and model definition.
Please make sure all input and output paths are correctly configured before running the scripts.
Depending on your local environment, you may need to modify file paths in the code.

Contact
If you have any questions or suggestions, please open an issue in this repository.
