#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

STEP=$1

if [ -z "$STEP" ]; then
  echo "Usage: ./run.sh [step1|step2|step3|step4|step5]"
  exit 1
fi

echo "============================================================"
echo "Running ${STEP}"
echo "Using config: ${SCRIPT_DIR}/config.sh"
echo "Device: ${DEVICE}"
echo "============================================================"

case "$STEP" in
  step1)
    CMD=(
      python "${SCRIPT_DIR}/run_pipeline.py"
      --step1
      --json_file "$JSON_FILE"
      --label_category_file "$LABEL_CATEGORY_FILE"
      --train_save_dir "$STEP1_TRAIN_SAVE_DIR"
      --test_save_dir "$STEP1_TEST_SAVE_DIR"
      --esm_batch_size "$STEP1_ESM_BATCH_SIZE"
      --train_max_length "$STEP1_TRAIN_MAX_LENGTH"
      --test_max_length "$STEP1_TEST_MAX_LENGTH"
      --device "$DEVICE"
    )

    if [ "$RUN_TRAIN_ESM" -eq 1 ]; then
      CMD+=(--run_train_esm)
    fi

    if [ "$RUN_TEST_ESM" -eq 1 ]; then
      CMD+=(--run_test_esm)
    fi

    "${CMD[@]}"
    ;;

  step2)
    CMD=(
      python "${SCRIPT_DIR}/run_pipeline.py"
      --step2
      --sae_checkpoint "$SAE_CHECKPOINT"
      --train_tensors "$STEP2_TRAIN_TENSORS"
      --test_tensors "$STEP2_TEST_TENSORS"
      --train_pt_dir "$STEP2_TRAIN_PT_DIR"
      --test_pt_dir "$STEP2_TEST_PT_DIR"
      --sae_batch_size "$STEP2_SAE_BATCH_SIZE"
      --save_every "$STEP2_SAVE_EVERY"
      --device "$DEVICE"
    )

    if [ "$RUN_TRAIN_SAE" -eq 1 ]; then
      CMD+=(--run_train_sae)
    fi

    if [ "$RUN_TEST_SAE" -eq 1 ]; then
      CMD+=(--run_test_sae)
    fi

    "${CMD[@]}"
    ;;

  step3)
    python "${SCRIPT_DIR}/run_pipeline.py" \
      --step3 \
      --train_tensors "$STEP3_TRAIN_TENSORS" \
      --test_tensors "$STEP3_TEST_TENSORS" \
      --train_pt_dir "$STEP3_TRAIN_PT_DIR" \
      --test_pt_dir "$STEP3_TEST_PT_DIR" \
      --label_mapping_file "$STEP3_LABEL_MAPPING_FILE" \
      --group_name "$STEP3_GROUP_NAME" \
      --max_iterations "$STEP3_MAX_ITERATIONS" \
      --final_feature_num "$STEP3_FINAL_FEATURE_NUM" \
      --lr "$STEP3_LR" \
      --base_lambda_l1 "$STEP3_BASE_LAMBDA_L1" \
      --step3_output_dir "$STEP3_OUTPUT_DIR" \
      --feature_dim "$FEATURE_DIM" \
      --device "$DEVICE"
    ;;

  step4)
    python "${SCRIPT_DIR}/run_pipeline.py" \
      --step4 \
      --test_tensors "$STEP4_TEST_TENSORS" \
      --test_pt_dir "$STEP4_TEST_PT_DIR" \
      --selected_features_dir "$STEP4_SELECTED_FEATURES_DIR" \
      --model_root "$STEP4_MODEL_ROOT" \
      --class_ids "${STEP4_CLASS_IDS[@]}" \
      --step4_output_dir "$STEP4_OUTPUT_DIR" \
      --feature_dim "$FEATURE_DIM" \
      --device "$DEVICE"
    ;;

  step5)
    python "${SCRIPT_DIR}/run_pipeline.py" \
      --step5 \
      --selected_features_csv "$STEP5_SELECTED_FEATURES_CSV" \
      --model_filepath "$STEP5_MODEL_FILEPATH" \
      --review_pt_folder "$STEP5_REVIEW_PT_FOLDER" \
      --train_auc_csv "$STEP5_TRAIN_AUC_CSV" \
      --review_tensors "$STEP5_REVIEW_TENSORS" \
      --review_batch_size "$STEP5_REVIEW_BATCH_SIZE" \
      --review_score_csv "$STEP5_REVIEW_SCORE_CSV" \
      --review_score_with_id_csv "$STEP5_REVIEW_SCORE_WITH_ID_CSV" \
      --feature_dim "$FEATURE_DIM" \
      --device "$DEVICE"
    ;;

  *)
    echo "Unknown step: $STEP"
    echo "Usage: ./run.sh [step1|step2|step3|step4|step5]"
    exit 1
    ;;
esac

echo "============================================================"
echo "${STEP} finished successfully."
echo "============================================================"