#!/bin/bash

# =============================================================================
# Global settings
# =============================================================================
DEVICE="cuda:1"
FEATURE_DIM=40960

# =============================================================================
# Step 1: ESM extraction
# =============================================================================
JSON_FILE="/mnt/data/fcc/binary_ESM/negtive_seg_select/mut_appended_sequences.json"
LABEL_CATEGORY_FILE="label_categories_uniprot_CDCODE_phasepdb_20250724.json"

STEP1_TRAIN_SAVE_DIR="./my_saved_tensors/train"
STEP1_TEST_SAVE_DIR="./my_saved_tensors/test"

STEP1_ESM_BATCH_SIZE=100
STEP1_TRAIN_MAX_LENGTH=3072
STEP1_TEST_MAX_LENGTH=2900

RUN_TRAIN_ESM=1
RUN_TEST_ESM=1

# =============================================================================
# Step 2: SAE sparsification
# =============================================================================
SAE_CHECKPOINT="step_80000.pt"

STEP2_TRAIN_TENSORS="./my_saved_tensors/train/train_tensors.pt"
STEP2_TEST_TENSORS="./my_saved_tensors/test/test_tensors.pt"

STEP2_TRAIN_PT_DIR="./protein_reps/train"
STEP2_TEST_PT_DIR="./protein_reps/test"

STEP2_SAE_BATCH_SIZE=64
STEP2_SAVE_EVERY=500

RUN_TRAIN_SAE=1
RUN_TEST_SAE=1

# =============================================================================
# Step 3: Train + test together
# =============================================================================
STEP3_TRAIN_TENSORS="./polyG/batch_full_results_20260521/my_saved_tensors/3072/train_tensors.pt"
STEP3_TEST_TENSORS="./my_saved_tensors/uniprot&CDCODE&phasepdb_20250724/3072/test_tensors.pt"

STEP3_TRAIN_PT_DIR="./polyG/batch_full_results_20260521/protein_reps_polyg_pt/train"
STEP3_TEST_PT_DIR="./protein_reps_uniprot&CDCODE&phasepdb_20250724_pt/test"

STEP3_LABEL_MAPPING_FILE="label_categories_proteins_with_PolyG_split.json"
STEP3_GROUP_NAME="PolyG"

STEP3_MAX_ITERATIONS=10
STEP3_FINAL_FEATURE_NUM=3000
STEP3_LR=0.0001
STEP3_BASE_LAMBDA_L1=0.000001

STEP3_OUTPUT_DIR="./final_test_results_class_20260522_PolyG"

# =============================================================================
# Step 4: Standalone evaluation
# =============================================================================
STEP4_TEST_TENSORS="./my_saved_tensors/uniprot&CDCODE&phasepdb_20250724/3072/test_tensors.pt"
STEP4_TEST_PT_DIR="./protein_reps_uniprot&CDCODE&phasepdb_20250724_pt/test"

STEP4_SELECTED_FEATURES_DIR="./final_results_20250724"
STEP4_MODEL_ROOT="./dynamic_feature_models/iteration_8"

# Multiple class IDs can be set here, separated by spaces
STEP4_CLASS_IDS=(13)

STEP4_OUTPUT_DIR="./step4_outputs"

# =============================================================================
# Step 5: Review protein prediction
# =============================================================================
STEP5_SELECTED_FEATURES_CSV="./final_results_20250724/selected_features_4.csv"
STEP5_MODEL_FILEPATH="./dynamic_feature_models/iteration_4/class_4"
STEP5_REVIEW_PT_FOLDER="./negtive_seg_select/protein_reps_mut_pt/test"
STEP5_TRAIN_AUC_CSV="./final_results_20250724/best_model_performance_4.csv"
STEP5_REVIEW_TENSORS="./negtive_seg_select/my_saved_tensors/3072/test_tensors.pt"

STEP5_REVIEW_BATCH_SIZE=64
STEP5_REVIEW_SCORE_CSV="final_protein_scores.csv"
STEP5_REVIEW_SCORE_WITH_ID_CSV="final_protein_scores_with_uniprot.csv"