#!/usr/bin/env python3
"""
Evaluation script for CE models trained with spatial feature concatenation.

This script evaluates models trained with run_ce_spatial_concat.py, which use
spatial feature concatenation to mimic reconstruction before classification.

Outputs:
1. Summary CSV with aggregated metrics per dataset
2. Detailed predictions CSV with per-sample (per-pair) predictions for error analysis

Metrics computed:
- Average Precision (AP)
- ROC AUC
- Precision, Recall, F1, Specificity, MCC (at optimal threshold)
- Brier score and Expected Calibration Error (ECE)
- Confusion matrix statistics
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(1, '../')
import helpers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss
)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve


class SpatialConcatClassifier(nn.Module):
    """
    Classifier that concatenates spatial features from lung pairs.

    Must match the architecture from run_ce_spatial_concat.py
    """
    def __init__(self, siamese_model, num_classes=2):
        super(SpatialConcatClassifier, self).__init__()
        self.siamese_model = siamese_model
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x1, x2):
        """Forward pass with spatial concatenation"""
        spatial1 = self.siamese_model.forward_one_spatial(x1)
        spatial2 = self.siamese_model.forward_one_spatial(x2)
        concatenated = torch.cat([spatial1, spatial2], dim=3)
        pooled = F.adaptive_avg_pool2d(concatenated, output_size=1)
        features = torch.flatten(pooled, 1)
        logits = self.classifier(features)
        return logits


def find_optimal_threshold_f1(y_true, y_pred_proba):
    """Find threshold that maximizes F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    return optimal_threshold


def compute_calibration_metrics(y_true, y_pred_proba, n_bins=10):
    """Compute calibration metrics"""
    try:
        brier = brier_score_loss(y_true, y_pred_proba)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

        return {
            'brier_score': float(brier),
            'ece': float(ece)
        }
    except:
        return {
            'brier_score': np.nan,
            'ece': np.nan
        }


def evaluate_predictions(y_true, y_pred_proba):
    """Compute comprehensive metrics"""
    metrics = {}

    try:
        metrics['ap'] = average_precision_score(y_true, y_pred_proba)
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['ap'] = np.nan
        metrics['auc'] = np.nan

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold_f1(y_true, y_pred_proba)
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)

    # Threshold-dependent metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    metrics['threshold'] = float(optimal_threshold)
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred_binary)
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)

    # Calibration
    cal_metrics = compute_calibration_metrics(y_true, y_pred_proba)
    metrics.update(cal_metrics)

    # Prediction statistics
    metrics['pred_mean'] = float(np.mean(y_pred_proba))
    metrics['pred_std'] = float(np.std(y_pred_proba))
    metrics['pos_ratio'] = float(np.mean(y_true))

    return metrics


def evaluate_spatial_concat_model(model, dataloader, device):
    """
    Evaluate spatial concat CE model

    Returns:
        probs: Predicted probabilities for positive class (per pair)
        labels: True labels (per pair)
        names: Sample names (pair identifiers)
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch - use pair labels
            img1, img2, label1, label2, pair_class_idx, path1, path2 = batch
            img1, img2 = img1.to(device), img2.to(device)

            # Get logits from spatial concatenation
            logits = model(img1, img2)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            probs_pos = probs[:, 1]  # Positive class (nodule) probability

            all_probs.extend(probs_pos.cpu().numpy())
            all_labels.extend(pair_class_idx.cpu().numpy())

            # Use first lung path as pair identifier
            all_names.extend(path1)

    return np.array(all_probs), np.array(all_labels), all_names


def build_transforms(resize_dim: int, single=False):
    """Build transforms for testing (no augmentation)"""
    if not single:
        transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])
    return transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CE models trained with spatial feature concatenation"
    )

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="../split_node21_sets")
    parser.add_argument("--label_csv", type=str, default="../split_node21_sets/only_nodule_half_labels.csv",
                       help="Path to CSV with individual lung labels")
    parser.add_argument("--process", type=str, required=True,
                       choices=["lung_seg", "crop", "arch_seg"])
    parser.add_argument("--train_source", type=str, required=True,
                       choices=["chestxray14", "jsrt", "padchest"])

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to evaluate")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                       help="Name for checkpoint in output (defaults to filename)")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["rgb", "grey", "single", "rad", "randinit"])
    parser.add_argument("--no_single_embedding", action="store_true",
                       help="Use 4D backbone output (B, 2048, 7, 7) - REQUIRED")
    parser.add_argument("--early_truncation", action="store_true",
                       help="Use layer3 truncation (B, 1024, 14, 14)")

    # Output arguments
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path to output summary CSV")
    parser.add_argument("--predictions_csv", type=str, default=None,
                       help="Path to output detailed predictions CSV (optional)")

    # Other arguments
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--resize_dim", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_in_ram", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Spatial concat requires spatial models
    if not args.no_single_embedding:
        raise ValueError("--no_single_embedding is REQUIRED for spatial concatenation evaluation")

    # Use checkpoint filename as name if not provided
    if args.checkpoint_name is None:
        args.checkpoint_name = Path(args.checkpoint).stem

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup paths
    external_test_names = ["chestxray14", "jsrt", "padchest"]
    test_path = os.path.join(args.dataset_path, args.process, args.train_source, "test")

    external_test_names.remove(args.train_source)
    test2_path = os.path.join(args.dataset_path, args.process, external_test_names[0], "test")
    test3_path = os.path.join(args.dataset_path, args.process, external_test_names[1], "test")

    # Build transform
    transform = build_transforms(args.resize_dim, single=(args.model_name == "single"))

    # Load dataloaders
    print("Loading test datasets...")
    dataloaders = {}

    test_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=test_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=False,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers,
        shuffle=False
    )
    dataloaders[f"test-{args.train_source}"] = test_dataloader

    test2_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=test2_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=False,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers,
        shuffle=False
    )
    dataloaders[f"test-{external_test_names[0]}"] = test2_dataloader

    test3_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=test3_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=False,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers,
        shuffle=False
    )
    dataloaders[f"test-{external_test_names[1]}"] = test3_dataloader

    print(f"Loaded {len(dataloaders)} test datasets")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")

    # Validate early_truncation requires no_single_embedding
    if args.early_truncation and not args.no_single_embedding:
        raise ValueError("--early_truncation requires --no_single_embedding flag")

    # Load backbone with appropriate truncation
    truncation_layer = 3 if args.early_truncation else 4
    backbone = helpers.models.load_truncated_model(
        args.model_name,
        single_embedding=not args.no_single_embedding,
        truncation_layer=truncation_layer
    )

    # Create Siamese model (no classification head needed)
    if args.early_truncation:
        siamese_model = helpers.models.SiameseNetworkSpatialEarly(
            backbone,
            embedding_dim=128,
            freeze_backbone=False,
            num_classes=None
        ).to(device)
    else:
        siamese_model = helpers.models.SiameseNetworkSpatial(
            backbone,
            embedding_dim=128,
            freeze_backbone=False,
            num_classes=None
        ).to(device)

    # Wrap with spatial concat classifier
    model = SpatialConcatClassifier(siamese_model, num_classes=2).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'eval_metrics' in checkpoint:
        print(f"Checkpoint eval metrics: {checkpoint['eval_metrics']}")

    # Evaluate on all datasets
    print(f"\n{'='*80}")
    print(f"Evaluating: {args.checkpoint_name}")
    print(f"{'='*80}")

    all_results = []
    all_predictions = []

    for ds_name, dataloader in dataloaders.items():
        print(f"\nEvaluating on {ds_name}...")

        # Get predictions
        probs, labels, names = evaluate_spatial_concat_model(model, dataloader, device)

        # Compute metrics
        metrics = evaluate_predictions(labels, probs)

        # Store results
        result = {
            'checkpoint': args.checkpoint_name,
            'dataset': ds_name,
            **metrics
        }
        all_results.append(result)

        # Print summary
        print(f"  {ds_name:25s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, "
              f"F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")
        print(f"  {'':25s}   Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
              f"Spec: {metrics['specificity']:.4f}, MCC: {metrics['mcc']:.4f}")

        # Save individual predictions if requested
        if args.predictions_csv:
            threshold = metrics['threshold']
            predictions = (probs >= threshold).astype(int)
            correct = (predictions == labels).astype(int)

            for i, name in enumerate(names):
                pred_row = {
                    'checkpoint': args.checkpoint_name,
                    'dataset': ds_name,
                    'sample_name': name,
                    'true_label': int(labels[i]),
                    'predicted_prob': float(probs[i]),
                    'predicted_label': int(predictions[i]),
                    'correct': int(correct[i]),
                    'threshold': float(threshold)
                }
                all_predictions.append(pred_row)

    # Save summary results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Summary results saved to: {args.output_csv}")

    # Save detailed predictions if requested
    if args.predictions_csv and all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(args.predictions_csv, index=False)
        print(f"✓ Detailed predictions saved to: {args.predictions_csv}")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
