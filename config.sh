#!/bin/bash

# =============================================================================
# Project root
# =============================================================================

DATASET_ROOT="./dataset"

# Use an absolute path if run.sh may be called from different directories.
# For example:
# DATASET_ROOT="/mnt/data/fcc/binary_ESM/dataset"

DEVICE="cuda:1"
FEATURE_DIM=40960


# =============================================================================
# Input files
# =============================================================================

INPUT_DIR="${DATASET_ROOT}/input"
CHECKPOINT_DIR="${DATASET_ROOT}/checkpoints"

SEQUENCE_JSON="${INPUT_DIR}/mut_appended_sequences.json"

LABEL_CATEGORY_FILE="${INPUT_DIR}/label_categories_uniprot_CDCODE_phasepdb_20250724.json"

LABEL_MAPPING_FILE="${INPUT_DIR}/label_categories_proteins_with_PolyG_split.json"

NEW_CONSTITUTIVE_SEQUENCE_JSON="${INPUT_DIR}/new_constitutive_sequences.json"

SAE_CHECKPOINT="${CHECKPOINT_DIR}/step_80000.pt"


# =============================================================================
# encode
#
# This stage combines:
#   1. ESM representation extraction
#   2. SAE sparse feature encoding
# =============================================================================

ENCODE_DIR="${DATASET_ROOT}/encoded"

ENCODE_TENSOR_DIR="${ENCODE_DIR}/tensors"
ENCODE_PROTEIN_REP_DIR="${ENCODE_DIR}/protein_reps"

ENCODE_TRAIN_TENSORS="${ENCODE_TENSOR_DIR}/train_tensors.pt"
ENCODE_TEST_TENSORS="${ENCODE_TENSOR_DIR}/test_tensors.pt"

ENCODE_TRAIN_PT_DIR="${ENCODE_PROTEIN_REP_DIR}/train"
ENCODE_TEST_PT_DIR="${ENCODE_PROTEIN_REP_DIR}/test"

ENCODE_NEW_CONSTITUTIVE_PT_DIR="${ENCODE_PROTEIN_REP_DIR}/new_constitutive"

ENCODE_ESM_BATCH_SIZE=100
ENCODE_SAE_BATCH_SIZE=64

ENCODE_TRAIN_MAX_LENGTH=3072
ENCODE_TEST_MAX_LENGTH=2900

ENCODE_SAVE_EVERY=500

RUN_TRAIN_ESM=1
RUN_TEST_ESM=1

RUN_TRAIN_SAE=1
RUN_TEST_SAE=1


# =============================================================================
# train+test
#
# Dynamic feature selection, classifier training, and test evaluation
# =============================================================================

TRAIN_TEST_DIR="${DATASET_ROOT}/train_test"

TRAIN_TEST_SELECTED_FEATURES_DIR="${TRAIN_TEST_DIR}/selected_features"
TRAIN_TEST_MODEL_DIR="${TRAIN_TEST_DIR}/models"
TRAIN_TEST_METRICS_DIR="${TRAIN_TEST_DIR}/metrics"

TRAIN_TEST_TRAIN_TENSORS="${ENCODE_TRAIN_TENSORS}"
TRAIN_TEST_TEST_TENSORS="${ENCODE_TEST_TENSORS}"

TRAIN_TEST_TRAIN_PT_DIR="${ENCODE_TRAIN_PT_DIR}"
TRAIN_TEST_TEST_PT_DIR="${ENCODE_TEST_PT_DIR}"

TRAIN_TEST_GROUP_NAME="PolyG"

TRAIN_TEST_MAX_ITERATIONS=10
TRAIN_TEST_FINAL_FEATURE_NUM=3000
TRAIN_TEST_LR=0.0001
TRAIN_TEST_BASE_LAMBDA_L1=0.000001

TRAIN_TEST_OUTPUT_DIR="${TRAIN_TEST_DIR}/outputs"


# =============================================================================
# test
#
# Standalone evaluation using saved selected features and trained models
# =============================================================================

TEST_DIR="${DATASET_ROOT}/test"

TEST_TENSORS="${ENCODE_TEST_TENSORS}"
TEST_PT_DIR="${ENCODE_TEST_PT_DIR}"

TEST_SELECTED_FEATURES_DIR="${TRAIN_TEST_SELECTED_FEATURES_DIR}"
TEST_MODEL_ROOT="${TRAIN_TEST_MODEL_DIR}"

TEST_CLASS_IDS=(13)

TEST_OUTPUT_DIR="${TEST_DIR}/outputs"


# =============================================================================
# new constitutive prediction
#
# Prediction for new constitutive proteins
# =============================================================================

PREDICTION_DIR="${DATASET_ROOT}/new_constitutive_prediction"
PREDICTION_OUTPUT_DIR="${PREDICTION_DIR}/outputs"

PREDICTION_SELECTED_FEATURES_CSV="${TRAIN_TEST_SELECTED_FEATURES_DIR}/selected_features_4.csv"

PREDICTION_MODEL_FILEPATH="${TRAIN_TEST_MODEL_DIR}/iteration_4/class_4"

PREDICTION_TRAIN_AUC_CSV="${TRAIN_TEST_METRICS_DIR}/best_model_performance_4.csv"

PREDICTION_PT_FOLDER="${ENCODE_NEW_CONSTITUTIVE_PT_DIR}"

PREDICTION_TENSORS="${PREDICTION_DIR}/tensors/test_tensors.pt"

PREDICTION_BATCH_SIZE=64

PREDICTION_SCORE_CSV="${PREDICTION_OUTPUT_DIR}/final_protein_scores.csv"

PREDICTION_SCORE_WITH_ID_CSV="${PREDICTION_OUTPUT_DIR}/final_protein_scores_with_uniprot.csv"