#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# Resolve project directory and load configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make paths such as ./dataset relative to the project directory.
cd "${SCRIPT_DIR}"

CONFIG_FILE="${SCRIPT_DIR}/config.sh"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: configuration file not found:"
    echo "  ${CONFIG_FILE}"
    exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

PYTHON_BIN="${PYTHON_BIN:-python}"


# =============================================================================
# Utility functions
# =============================================================================

usage() {
    cat <<EOF
Usage:
  ./run.sh encode
  ./run.sh train_test
  ./run.sh train+test
  ./run.sh test
  ./run.sh predict

Stages:
  encode       Run ESM extraction followed by SAE encoding.
  train_test   Train dynamic feature models and evaluate on the test set.
  train+test   Alias for train_test.
  test         Standalone evaluation using saved models and features.
  predict      Predict scores for new constitutive proteins.
EOF
}

require_file() {
    local file_path="$1"

    if [[ ! -f "${file_path}" ]]; then
        echo "Error: required file does not exist:"
        echo "  ${file_path}"
        exit 1
    fi
}

require_dir() {
    local dir_path="$1"

    if [[ ! -d "${dir_path}" ]]; then
        echo "Error: required directory does not exist:"
        echo "  ${dir_path}"
        exit 1
    fi
}

print_header() {
    local stage="$1"

    echo
    echo "============================================================"
    echo "Stage: ${stage}"
    echo "Project: ${SCRIPT_DIR}"
    echo "Dataset: ${DATASET_ROOT}"
    echo "Device:  ${DEVICE}"
    echo "============================================================"
    echo
}

run_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'

    "$@"
}


# =============================================================================
# Stage: encode
#
# Original Step 1:
#   ESM representation extraction
#
# Original Step 2:
#   SAE sparse feature encoding
# =============================================================================

run_encode() {
    print_header "encode"

    # The original Step 1 writes train_tensors.pt and test_tensors.pt
    # into these two directories.
    local encode_train_save_dir="${ENCODE_TENSOR_DIR}/train"
    local encode_test_save_dir="${ENCODE_TENSOR_DIR}/test"

    local encode_train_tensors="${ENCODE_TRAIN_TENSORS:-${encode_train_save_dir}/train_tensors.pt}"
    local encode_test_tensors="${ENCODE_TEST_TENSORS:-${encode_test_save_dir}/test_tensors.pt}"

    mkdir -p \
        "${encode_train_save_dir}" \
        "${encode_test_save_dir}" \
        "${ENCODE_TRAIN_PT_DIR}" \
        "${ENCODE_TEST_PT_DIR}"

    require_file "${SEQUENCE_JSON}"
    require_file "${LABEL_CATEGORY_FILE}"
    require_file "${SAE_CHECKPOINT}"

    echo "[1/2] Extracting ESM representations"

    local esm_command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py"
        --step1
        --json_file "${SEQUENCE_JSON}"
        --label_category_file "${LABEL_CATEGORY_FILE}"
        --train_save_dir "${encode_train_save_dir}"
        --test_save_dir "${encode_test_save_dir}"
        --esm_batch_size "${ENCODE_ESM_BATCH_SIZE}"
        --train_max_length "${ENCODE_TRAIN_MAX_LENGTH}"
        --test_max_length "${ENCODE_TEST_MAX_LENGTH}"
        --device "${DEVICE}"
    )

    if [[ "${RUN_TRAIN_ESM}" -eq 1 ]]; then
        esm_command+=(--run_train_esm)
    fi

    if [[ "${RUN_TEST_ESM}" -eq 1 ]]; then
        esm_command+=(--run_test_esm)
    fi

    run_command "${esm_command[@]}"

    require_file "${encode_train_tensors}"
    require_file "${encode_test_tensors}"

    echo
    echo "[2/2] Encoding representations with SAE"

    local sae_command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py"
        --step2
        --sae_checkpoint "${SAE_CHECKPOINT}"
        --train_tensors "${encode_train_tensors}"
        --test_tensors "${encode_test_tensors}"
        --train_pt_dir "${ENCODE_TRAIN_PT_DIR}"
        --test_pt_dir "${ENCODE_TEST_PT_DIR}"
        --sae_batch_size "${ENCODE_SAE_BATCH_SIZE}"
        --save_every "${ENCODE_SAVE_EVERY}"
        --device "${DEVICE}"
    )

    if [[ "${RUN_TRAIN_SAE}" -eq 1 ]]; then
        sae_command+=(--run_train_sae)
    fi

    if [[ "${RUN_TEST_SAE}" -eq 1 ]]; then
        sae_command+=(--run_test_sae)
    fi

    run_command "${sae_command[@]}"

    echo
    echo "encode completed successfully."
}


# =============================================================================
# Stage: train_test
#
# Original Step 3:
#   Dynamic feature selection, classifier training, and test evaluation
# =============================================================================

run_train_test() {
    print_header "train+test"

    mkdir -p \
        "${TRAIN_TEST_OUTPUT_DIR}" \
        "${TRAIN_TEST_SELECTED_FEATURES_DIR}" \
        "${TRAIN_TEST_MODEL_DIR}" \
        "${TRAIN_TEST_METRICS_DIR}"

    require_file "${TRAIN_TEST_TRAIN_TENSORS}"
    require_file "${TRAIN_TEST_TEST_TENSORS}"
    require_dir "${TRAIN_TEST_TRAIN_PT_DIR}"
    require_dir "${TRAIN_TEST_TEST_PT_DIR}"
    require_file "${LABEL_MAPPING_FILE}"

    local train_test_command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py"
        --step3
        --train_tensors "${TRAIN_TEST_TRAIN_TENSORS}"
        --test_tensors "${TRAIN_TEST_TEST_TENSORS}"
        --train_pt_dir "${TRAIN_TEST_TRAIN_PT_DIR}"
        --test_pt_dir "${TRAIN_TEST_TEST_PT_DIR}"
        --label_mapping_file "${LABEL_MAPPING_FILE}"
        --group_name "${TRAIN_TEST_GROUP_NAME}"
        --max_iterations "${TRAIN_TEST_MAX_ITERATIONS}"
        --final_feature_num "${TRAIN_TEST_FINAL_FEATURE_NUM}"
        --lr "${TRAIN_TEST_LR}"
        --base_lambda_l1 "${TRAIN_TEST_BASE_LAMBDA_L1}"
        --step3_output_dir "${TRAIN_TEST_OUTPUT_DIR}"
        --device "${DEVICE}"
    )

    # Add this only if run_pipeline.py defines --feature_dim for Step 3.
    if [[ "${TRAIN_TEST_PASS_FEATURE_DIM:-0}" -eq 1 ]]; then
        train_test_command+=(--feature_dim "${FEATURE_DIM}")
    fi

    run_command "${train_test_command[@]}"

    echo
    echo "train+test completed successfully."
}


# =============================================================================
# Stage: test
#
# Original Step 4:
#   Standalone evaluation using saved selected features and models
# =============================================================================

run_test() {
    print_header "test"

    mkdir -p "${TEST_OUTPUT_DIR}"

    require_file "${TEST_TENSORS}"
    require_dir "${TEST_PT_DIR}"
    require_dir "${TEST_SELECTED_FEATURES_DIR}"
    require_dir "${TEST_MODEL_ROOT}"

    if [[ "${#TEST_CLASS_IDS[@]}" -eq 0 ]]; then
        echo "Error: TEST_CLASS_IDS is empty."
        exit 1
    fi

    local test_command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py"
        --step4
        --test_tensors "${TEST_TENSORS}"
        --test_pt_dir "${TEST_PT_DIR}"
        --selected_features_dir "${TEST_SELECTED_FEATURES_DIR}"
        --model_root "${TEST_MODEL_ROOT}"
        --class_ids "${TEST_CLASS_IDS[@]}"
        --step4_output_dir "${TEST_OUTPUT_DIR}"
        --feature_dim "${FEATURE_DIM}"
        --device "${DEVICE}"
    )

    run_command "${test_command[@]}"

    echo
    echo "test completed successfully."
}


# =============================================================================
# Stage: predict
#
# Original Step 5:
#   Prediction for new constitutive proteins
# =============================================================================

run_predict() {
    print_header "new constitutive prediction"

    mkdir -p "${PREDICTION_OUTPUT_DIR}"

    require_file "${PREDICTION_SELECTED_FEATURES_CSV}"
    require_dir "${PREDICTION_MODEL_FILEPATH}"
    require_dir "${PREDICTION_PT_FOLDER}"
    require_file "${PREDICTION_TRAIN_AUC_CSV}"
    require_file "${PREDICTION_TENSORS}"

    local prediction_command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py"
        --step5
        --selected_features_csv "${PREDICTION_SELECTED_FEATURES_CSV}"
        --model_filepath "${PREDICTION_MODEL_FILEPATH}"
        --review_pt_folder "${PREDICTION_PT_FOLDER}"
        --train_auc_csv "${PREDICTION_TRAIN_AUC_CSV}"
        --review_tensors "${PREDICTION_TENSORS}"
        --review_batch_size "${PREDICTION_BATCH_SIZE}"
        --review_score_csv "${PREDICTION_SCORE_CSV}"
        --review_score_with_id_csv "${PREDICTION_SCORE_WITH_ID_CSV}"
        --feature_dim "${FEATURE_DIM}"
        --device "${DEVICE}"
    )

    run_command "${prediction_command[@]}"

    echo
    echo "new constitutive prediction completed successfully."
}


# =============================================================================
# Main entry point
# =============================================================================

if [[ "$#" -ne 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    encode)
        run_encode
        ;;

    train_test|train+test)
        run_train_test
        ;;

    test)
        run_test
        ;;

    predict)
        run_predict
        ;;

    -h|--help|help)
        usage
        ;;

    *)
        echo "Error: unknown stage '$1'"
        echo
        usage
        exit 1
        ;;
esac

echo
echo "============================================================"
echo "Stage '$1' finished successfully."
echo "============================================================"