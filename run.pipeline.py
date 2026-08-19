#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import esm

from data_prepare import load_train_set, load_test_set
from dictionary import AutoEncoder
from sae_classify import (
    process_in_batches,
    process_and_save_sae_features,
    save_label_categories,
    save_best_feature_selection_results,
    save_scores_to_csv,
)
from train_classify import (
    train_dynamic_feature_selection,
    evaluate_all_1vs1_models_on_test,
    evaluate_all_1vs1_models_on_test_from_pth,
    evaluate_all_1vs1_models_on_test_to_new_prots,
)

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path):
    """Create a directory if it does not exist."""
    if path is not None:
        os.makedirs(path, exist_ok=True)


def build_binary_mask(feature_indices, feature_dim):
    """Build a binary mask tensor from selected feature indices."""
    binary_mask = torch.zeros(feature_dim, dtype=torch.int)
    for index in feature_indices:
        if 0 <= index < feature_dim:
            binary_mask[index] = 1
    return binary_mask


def load_selected_feature_mask(csv_file_path, feature_dim):
    """Load selected feature indices from CSV and convert them into a binary mask."""
    df = pd.read_csv(csv_file_path)
    if "feature_index" not in df.columns:
        raise ValueError(f"'feature_index' column not found in {csv_file_path}")
    feature_indices = df["feature_index"].astype(int).tolist()
    return build_binary_mask(feature_indices, feature_dim)


def load_auc_dict(csv_file_path):
    """Load class pair AUC scores from a CSV file."""
    auc_data = pd.read_csv(csv_file_path)
    if "class_pair" not in auc_data.columns or "roc_auc" not in auc_data.columns:
        raise ValueError(f"'class_pair' or 'roc_auc' column not found in {csv_file_path}")
    return auc_data[["class_pair", "roc_auc"]].set_index("class_pair").to_dict()["roc_auc"]


def save_evaluation_summary(results, output_csv):
    """Save evaluation summary results to a CSV file."""
    rows = []

    for (class_a, class_b), metrics in results["evaluation_summary"].items():
        rows.append({
            "class_pair": f"{class_a}_vs_{class_b}",
            "roc_auc": metrics["AUC"],
            "ap": metrics["AP"],
            "accuracy": metrics["Accuracy"],
            "f1_score": metrics["F1"],
        })

    rows.append({
        "class_pair": "average",
        "roc_auc": results["avg_auc"],
        "ap": results["avg_ap"],
        "accuracy": np.mean([m["Accuracy"] for m in results["evaluation_summary"].values()]),
        "f1_score": results["avg_f1"],
    })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)


def print_evaluation_summary(results):
    """Print detailed evaluation summary."""
    if not results:
        print("No evaluation results returned.")
        return

    for (cls_a, cls_b), metrics in results["evaluation_summary"].items():
        print(
            f"{cls_a} vs {cls_b} -> "
            f"AUC: {metrics['AUC']:.4f} | "
            f"AP: {metrics['AP']:.4f} | "
            f"Acc: {metrics['Accuracy']:.4f} | "
            f"F1: {metrics['F1']:.4f}"
        )

    print("\nOverall average results:")
    print(f"  AUC: {results['avg_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"  AP:  {results['avg_ap']:.4f} ± {results['std_ap']:.4f}")
    print(f"  F1:  {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")


# =============================================================================
# Step 1: Extract ESM representations from raw sequences
# =============================================================================

def run_step1(args):
    """Run Step 1: extract ESM representations and save tensor files."""
    print("\n================ Step 1: ESM Extraction ================\n")

    print("Loading PLM model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    print("PLM model loaded successfully.\n")

    print(f"Loading sequence data from: {args.json_file}")
    sequences_train = load_train_set(args.json_file)
    sequences_test = load_test_set(args.json_file)

    print(f"Saving label category mapping to: {args.label_category_file}")
    save_label_categories(sequences_train, filepath=args.label_category_file)

    if args.run_train_esm:
        print("\nProcessing training set...")
        process_in_batches(
            sequences_train,
            model,
            batch_converter,
            alphabet,
            save_dir=args.train_save_dir,
            prefix="train",
            batch_size=args.esm_batch_size,
            max_length=args.train_max_length
        )

    if args.run_test_esm:
        print("\nProcessing test set...")
        process_in_batches(
            sequences_test,
            model,
            batch_converter,
            alphabet,
            save_dir=args.test_save_dir,
            prefix="test",
            batch_size=args.esm_batch_size,
            max_length=args.test_max_length
        )

    print("\nStep 1 completed.")


# =============================================================================
# Step 2: Apply SAE to saved ESM representations
# =============================================================================

def run_step2(args):
    """Run Step 2: apply SAE to PLM representations and save sparse features."""
    print("\n================ Step 2: SAE Sparsification ================\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading SAE checkpoint from: {args.sae_checkpoint}")
    sae = AutoEncoder.from_pretrained(args.sae_checkpoint)
    sae.eval()
    sae = sae.to(device)

    if args.run_train_sae:
        print(f"\nLoading training tensors from: {args.train_tensors}")
        train_data = torch.load(args.train_tensors)
        train_plm_rep = train_data["residue_representations"]

        print(f"Saving sparse training features to: {args.train_pt_dir}")
        process_and_save_sae_features(
            protein_list=train_plm_rep,
            sae_model=sae,
            output_dir=args.train_pt_dir,
            batch_size=args.sae_batch_size,
            save_every=args.save_every
        )

    if args.run_test_sae:
        print(f"\nLoading test tensors from: {args.test_tensors}")
        test_data = torch.load(args.test_tensors)
        test_plm_rep = test_data["residue_representations"]

        print(f"Saving sparse test features to: {args.test_pt_dir}")
        process_and_save_sae_features(
            protein_list=test_plm_rep,
            sae_model=sae,
            output_dir=args.test_pt_dir,
            batch_size=args.sae_batch_size,
            save_every=args.save_every
        )

    print("\nStep 2 completed.")


# =============================================================================
# Step 3: Train + test together
# =============================================================================

def run_step3(args):
    """Run Step 3: train feature selection models and immediately evaluate them."""
    print("\n================ Step 3: Train + Evaluate ================\n")

    print(f"Loading training tensors from: {args.train_tensors}")
    train_data = torch.load(args.train_tensors)
    train_label_matrix = train_data["label_matrix"]

    print(f"Loading test tensors from: {args.test_tensors}")
    test_data = torch.load(args.test_tensors)
    test_label_matrix = test_data["label_matrix"]

    print(f"Loading label mapping from: {args.label_mapping_file}")
    with open(args.label_mapping_file, "r") as f:
        label_map = json.load(f)

    if args.group_name not in label_map:
        raise ValueError(f"group_name '{args.group_name}' not found in {args.label_mapping_file}")

    class_idx = label_map[args.group_name]
    print(f"group_name: {args.group_name}")
    print(f"class_idx: {class_idx}")

    print("Starting dynamic feature selection training...")
    iteration_results = train_dynamic_feature_selection(
        pt_folder=args.train_pt_dir,
        label_matrix=train_label_matrix,
        max_iterations=args.max_iterations,
        final_feature_num=args.final_feature_num,
        class_idx=class_idx,
        lr=args.lr,
        base_lambda_l1=args.base_lambda_l1
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nTraining finished. Starting test evaluation...")
    test_results_1vs1 = evaluate_all_1vs1_models_on_test(
        pt_folder=args.test_pt_dir,
        label_matrix_test=test_label_matrix,
        best_feature_mask=iteration_results["best_feature_masks"],
        binary_results=iteration_results["best_results"]["binary_results"],
        input_dim=args.feature_dim,
        device=args.device
    )

    for (cls_a, cls_b), metrics in test_results_1vs1.items():
        print(f"{cls_a} vs {cls_b} -> AUC: {metrics['AUC']:.4f}")

    if args.step3_output_dir is not None:
        ensure_dir(args.step3_output_dir)
        save_best_feature_selection_results(
            best_results={
                "best_iteration": iteration_results["best_iteration"],
                "best_performance": iteration_results["best_performance"],
                "best_feature_masks": iteration_results["best_feature_masks"],
                "best_results": iteration_results["best_results"],
                "best_feature_importances": iteration_results["best_feature_importances"],
                "test_results": test_results_1vs1
            },
            output_dir=args.step3_output_dir
        )
        print(f"Saved Step 3 results to: {args.step3_output_dir}")

    print("\nStep 3 completed.")


# =============================================================================
# Step 4: Evaluate using existing saved models and selected features
# =============================================================================

def run_step4(args):
    """Run Step 4: evaluate using existing model files and selected feature CSV files."""
    print("\n================ Step 4: Standalone Evaluation ================\n")

    print(f"Loading test tensors from: {args.test_tensors}")
    test_data = torch.load(args.test_tensors)
    test_label_matrix = test_data["label_matrix"]

    ensure_dir(args.step4_output_dir)

    for class_id in args.class_ids:
        print(f"\nProcessing class_id={class_id}")

        csv_file_path = os.path.join(args.selected_features_dir, f"selected_features_{class_id}.csv")
        model_folder = os.path.join(args.model_root, f"class_{class_id}")
        output_csv = os.path.join(args.step4_output_dir, f"best_model_performance_{class_id}.csv")

        print(f"Loading selected features from: {csv_file_path}")
        binary_mask = load_selected_feature_mask(csv_file_path, args.feature_dim)

        print(f"Evaluating models from: {model_folder}")
        results = evaluate_all_1vs1_models_on_test_from_pth(
            pt_folder=args.test_pt_dir,
            label_matrix_test=test_label_matrix,
            best_feature_mask=binary_mask,
            model_folder=model_folder,
            input_dim=args.feature_dim,
            device=args.device
        )

        if results:
            print_evaluation_summary(results)
            save_evaluation_summary(results, output_csv)
            print(f"Saved evaluation CSV to: {output_csv}")
        else:
            print(f"No results returned for class_id={class_id}")

    print("\nStep 4 completed.")


# =============================================================================
# Step 5: Predict scores for review proteins
# =============================================================================

def run_step5(args):
    """Run Step 5: predict scores for review proteins."""
    print("\n================ Step 5: Review Protein Prediction ================\n")

    print(f"Loading selected features from: {args.selected_features_csv}")
    binary_mask = load_selected_feature_mask(args.selected_features_csv, args.feature_dim)

    print(f"Loading training AUC file from: {args.train_auc_csv}")
    train_auc = load_auc_dict(args.train_auc_csv)

    print("Running prediction on review proteins...")
    evaluation_results = evaluate_all_1vs1_models_on_test_to_new_prots(
        model_filepath=args.model_filepath,
        pt_folder=args.review_pt_folder,
        best_feature_mask=binary_mask,
        train_auc=train_auc,
        input_dim=args.feature_dim,
        batch_size=args.review_batch_size,
        device=args.device
    )

    print(f"Saving raw review scores to: {args.review_score_csv}")
    save_scores_to_csv(evaluation_results, args.review_score_csv)

    print(f"Loading review tensor file from: {args.review_tensors}")
    review_data = torch.load(args.review_tensors)
    review_seq_ids = review_data["seq_ids"]

    df_scores = pd.read_csv(args.review_score_csv)
    if len(df_scores) != len(review_seq_ids):
        raise ValueError(
            "Length of review_seq_ids does not match the number of rows in the score CSV."
        )

    df_scores["UniProt_ID"] = review_seq_ids
    df_scores.to_csv(args.review_score_with_id_csv, index=False)

    print(f"Saved review scores with UniProt_ID to: {args.review_score_with_id_csv}")
    print("\nStep 5 completed.")


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Protein pipeline with 5 runnable steps")

    # Global arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--feature_dim", type=int, default=40960, help="Feature dimension")

    # Step switches
    parser.add_argument("--step1", action="store_true", help="Run Step 1: ESM extraction")
    parser.add_argument("--step2", action="store_true", help="Run Step 2: SAE sparsification")
    parser.add_argument("--step3", action="store_true", help="Run Step 3: Train + test")
    parser.add_argument("--step4", action="store_true", help="Run Step 4: Standalone evaluation")
    parser.add_argument("--step5", action="store_true", help="Run Step 5: Review protein prediction")

    # Step 1 arguments
    parser.add_argument("--json_file", type=str, help="Input JSON file containing sequences")
    parser.add_argument("--label_category_file", type=str, help="Output JSON for label categories")
    parser.add_argument("--train_save_dir", type=str, help="Directory to save Step 1 train tensors")
    parser.add_argument("--test_save_dir", type=str, help="Directory to save Step 1 test tensors")
    parser.add_argument("--esm_batch_size", type=int, default=100, help="Batch size for ESM extraction")
    parser.add_argument("--train_max_length", type=int, default=3072, help="Max length for training sequences")
    parser.add_argument("--test_max_length", type=int, default=2900, help="Max length for test sequences")
    parser.add_argument("--run_train_esm", action="store_true", help="Process training set in Step 1")
    parser.add_argument("--run_test_esm", action="store_true", help="Process test set in Step 1")

    # Step 2 arguments
    parser.add_argument("--sae_checkpoint", type=str, help="Path to SAE checkpoint")
    parser.add_argument("--train_tensors", type=str, help="Path to train_tensors.pt")
    parser.add_argument("--test_tensors", type=str, help="Path to test_tensors.pt")
    parser.add_argument("--train_pt_dir", type=str, help="Output directory for sparse train pt files")
    parser.add_argument("--test_pt_dir", type=str, help="Output directory for sparse test pt files")
    parser.add_argument("--sae_batch_size", type=int, default=64, help="Batch size for SAE processing")
    parser.add_argument("--save_every", type=int, default=500, help="Save frequency for SAE processing")
    parser.add_argument("--run_train_sae", action="store_true", help="Process training set in Step 2")
    parser.add_argument("--run_test_sae", action="store_true", help="Process test set in Step 2")

    # Step 3 arguments
    parser.add_argument("--label_mapping_file", type=str, help="JSON file containing label mapping")
    parser.add_argument("--group_name", type=str, help="Target group name for training")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum training iterations")
    parser.add_argument("--final_feature_num", type=int, default=3000, help="Final selected feature number")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--base_lambda_l1", type=float, default=1e-6, help="Base L1 regularization")
    parser.add_argument("--step3_output_dir", type=str, help="Output directory for Step 3 results")

    # Step 4 arguments
    parser.add_argument("--selected_features_dir", type=str, help="Directory containing selected_features_{class_id}.csv")
    parser.add_argument("--model_root", type=str, help="Root directory containing saved class models")
    parser.add_argument("--class_ids", type=int, nargs="*", default=[], help="Class IDs for Step 4 evaluation")
    parser.add_argument("--step4_output_dir", type=str, help="Output directory for Step 4 CSV results")

    # Step 5 arguments
    parser.add_argument("--selected_features_csv", type=str, help="Selected feature CSV for review prediction")
    parser.add_argument("--model_filepath", type=str, help="Model folder for review prediction")
    parser.add_argument("--review_pt_folder", type=str, help="PT folder for review proteins")
    parser.add_argument("--train_auc_csv", type=str, help="Training AUC CSV")
    parser.add_argument("--review_tensors", type=str, help="Tensor file containing review seq_ids")
    parser.add_argument("--review_batch_size", type=int, default=64, help="Batch size for review prediction")
    parser.add_argument("--review_score_csv", type=str, default="final_protein_scores.csv", help="Output score CSV")
    parser.add_argument(
        "--review_score_with_id_csv",
        type=str,
        default="final_protein_scores_with_uniprot.csv",
        help="Output score CSV with UniProt IDs"
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    if not any([args.step1, args.step2, args.step3, args.step4, args.step5]):
        raise ValueError("Please specify at least one step to run: --step1/--step2/--step3/--step4/--step5")

    if args.step1:
        run_step1(args)

    if args.step2:
        run_step2(args)

    if args.step3:
        run_step3(args)

    if args.step4:
        run_step4(args)

    if args.step5:
        run_step5(args)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()