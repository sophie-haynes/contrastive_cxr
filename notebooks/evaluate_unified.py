#!/usr/bin/env python3
"""
Unified evaluation framework that supports all model types:
1. Paired lung models (Individual SupCon, SupCon Pair, Hierarchical, etc.)
2. Reconstructed full CXR models (SupCon on full images)
3. Cross-entropy baseline models

Outputs:
1. Summary CSV with aggregated metrics (same format as individual eval scripts)
2. Detailed predictions CSV with per-sample predictions for error analysis

Evaluation methods:
- Distance-based classification (threshold on cosine distance)
- Linear probe (logistic regression on embeddings)
- Shallow MLP probe (single hidden layer)
- CE baseline (for end-to-end trained models)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(1, '../')
import helpers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
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


class ReconstructedCXRDataset(torch.utils.data.Dataset):
    """
    Dataset that reconstructs full CXR images from paired lung data.
    Returns format compatible with paired lung dataloaders for unified handling.
    """
    def __init__(self, paired_dataset, transform):
        self.paired_dataset = paired_dataset
        self.transform = transform
        self.pairs = paired_dataset.pairs
        self.cache_in_ram = paired_dataset.cache_in_ram
        self._image_cache = paired_dataset._image_cache if paired_dataset.cache_in_ram else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_info = self.pairs[idx]

        # Load lung images
        if self.cache_in_ram and self._image_cache:
            if pair_info['lungl_path'] in self._image_cache:
                lung_l = self._image_cache[pair_info['lungl_path']].copy()
            else:
                with Image.open(pair_info['lungl_path']) as im:
                    lung_l = im.convert('RGB')

            if pair_info['lungr_path'] in self._image_cache:
                lung_r = self._image_cache[pair_info['lungr_path']].copy()
            else:
                with Image.open(pair_info['lungr_path']) as im:
                    lung_r = im.convert('RGB')
        else:
            with Image.open(pair_info['lungl_path']) as im:
                lung_l = im.convert('RGB')
            with Image.open(pair_info['lungr_path']) as im:
                lung_r = im.convert('RGB')

        # Horizontally flip lung_r
        lung_r_flipped = lung_r.transpose(Image.FLIP_LEFT_RIGHT)

        # Concatenate
        width_l, height_l = lung_l.size
        width_r, height_r = lung_r_flipped.size

        if height_l != height_r:
            target_height = max(height_l, height_r)
            if height_l != target_height:
                lung_l = lung_l.resize((width_l, target_height), Image.BILINEAR)
            if height_r != target_height:
                lung_r_flipped = lung_r_flipped.resize((width_r, target_height), Image.BILINEAR)
            height_l = height_r = target_height

        reconstructed_cxr = Image.new('RGB', (width_l + width_r, height_l))
        reconstructed_cxr.paste(lung_l, (0, 0))
        reconstructed_cxr.paste(lung_r_flipped, (width_l, 0))

        image = self.transform(reconstructed_cxr)
        label = pair_info['class_idx']
        pair_name = pair_info['pair_name']

        # Return format: (image, image, label, name, name) - compatible with paired format
        return image, image, label, pair_name, pair_name


# ============================================================================
# Distance Classifiers
# ============================================================================

class DistanceClassifierPaired:
    """Distance-based classifier for paired lung models"""
    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder
        self.device = device
        self.optimal_threshold = None
        self.has_projection = hasattr(encoder, 'projection_head')

    def fit(self, dataloader):
        """Find optimal threshold on training data"""
        distances, labels, _ = self._get_distances_labels_names(dataloader)

        # Higher distance = positive class (for individual label SupCon)
        scores = distances
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.0

        return self

    def predict_proba_with_names(self, dataloader):
        """Get probabilities with sample names"""
        distances, labels, names = self._get_distances_labels_names(dataloader)
        probs = np.clip(distances, 0, 1)
        return probs, labels, names

    def _get_distances_labels_names(self, dataloader):
        """Extract distances, labels, and names"""
        self.encoder.eval()
        distances = []
        labels = []
        names = []

        with torch.no_grad():
            for batch in dataloader:
                x1, x2, label, name1, _ = batch
                x1, x2 = x1.to(self.device), x2.to(self.device)

                # Get embeddings
                if self.has_projection:
                    emb1, emb2 = self.encoder(x1, x2, return_embedding=True)
                else:
                    emb1, emb2 = self.encoder(x1, x2)

                # Cosine distance
                cosine_sim = torch.sum(emb1 * emb2, dim=1)
                dist = 1 - cosine_sim

                distances.extend(dist.cpu().numpy())
                labels.extend(label.numpy())
                names.extend(name1)

        return np.array(distances), np.array(labels), names


class DistanceClassifierFullImage:
    """Distance-based classifier for full image models (prototype-based)"""
    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder
        self.device = device
        self.optimal_threshold = None
        self.class_prototypes = None
        self.has_projection = hasattr(encoder, 'projection_head')

    def fit(self, dataloader):
        """Find class prototypes and optimal threshold"""
        embeddings, labels, _ = self._get_embeddings_labels_names(dataloader)

        # Compute class prototypes
        unique_labels = np.unique(labels)
        self.class_prototypes = {}
        for label in unique_labels:
            mask = labels == label
            self.class_prototypes[label] = embeddings[mask].mean(axis=0)

        # Compute distances to positive prototype
        pos_prototype = self.class_prototypes[1]
        distances = []
        for emb in embeddings:
            cosine_sim = np.dot(emb, pos_prototype) / (np.linalg.norm(emb) * np.linalg.norm(pos_prototype))
            dist = 1 - cosine_sim
            distances.append(dist)

        distances = np.array(distances)
        scores = -distances  # Negative distance as score (lower distance = positive)

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.0

        return self

    def predict_proba_with_names(self, dataloader):
        """Get probabilities with sample names"""
        embeddings, labels, names = self._get_embeddings_labels_names(dataloader)

        pos_prototype = self.class_prototypes[1]
        distances = []
        for emb in embeddings:
            cosine_sim = np.dot(emb, pos_prototype) / (np.linalg.norm(emb) * np.linalg.norm(pos_prototype) + 1e-8)
            dist = 1 - cosine_sim
            distances.append(dist)

        distances = np.array(distances)
        scores = -distances
        probs = 1 / (1 + np.exp(-scores))  # Sigmoid to get probabilities

        return probs, labels, names

    def _get_embeddings_labels_names(self, dataloader):
        """Extract embeddings, labels, and names"""
        self.encoder.eval()
        all_embeddings = []
        all_labels = []
        all_names = []

        with torch.no_grad():
            for batch in dataloader:
                x1, _, label, name, _ = batch  # x1 and x2 are same for reconstructed
                x1 = x1.to(self.device)

                # Get embedding
                if self.has_projection:
                    emb = self.encoder.forward_one(x1, return_embedding=True)
                else:
                    emb = self.encoder.forward_one(x1)

                all_embeddings.append(emb.cpu().numpy())
                all_labels.extend(label.numpy())
                all_names.extend(name)

        return np.vstack(all_embeddings), np.array(all_labels), all_names


# ============================================================================
# Probe Classifiers
# ============================================================================

class LinearProbe(nn.Module):
    """Linear classifier - works for both paired and full image"""
    def __init__(self, encoder, embedding_dim=128, paired_mode=True):
        super(LinearProbe, self).__init__()
        self.encoder = encoder
        self.paired_mode = paired_mode
        self.has_projection = hasattr(encoder, 'projection_head')

        for param in self.encoder.parameters():
            param.requires_grad = False

        if paired_mode:
            input_dim = 2 * embedding_dim + 1
            self.classifier = nn.Linear(input_dim, 1)
        else:
            self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x1, x2=None, distance_metric='cosine'):
        if self.paired_mode:
            if self.has_projection:
                emb1, emb2 = self.encoder(x1, x2, return_embedding=True)
            else:
                emb1, emb2 = self.encoder(x1, x2)

            if distance_metric == 'euclidean':
                distance = F.pairwise_distance(emb1, emb2, p=2)
            else:
                cosine_sim = torch.sum(emb1 * emb2, dim=1)
                distance = 1 - cosine_sim

            combined = torch.cat([emb1, emb2, distance.unsqueeze(1)], dim=1)
            logits = self.classifier(combined).squeeze()
        else:
            if self.has_projection:
                emb = self.encoder.forward_one(x1, return_embedding=True)
            else:
                emb = self.encoder.forward_one(x1)

            logits = self.classifier(emb).squeeze()

        return logits


class ShallowMLPProbe(nn.Module):
    """Shallow MLP classifier - works for both paired and full image"""
    def __init__(self, encoder, embedding_dim=128, hidden_dim=64, dropout=0.3, paired_mode=True):
        super(ShallowMLPProbe, self).__init__()
        self.encoder = encoder
        self.paired_mode = paired_mode
        self.has_projection = hasattr(encoder, 'projection_head')

        for param in self.encoder.parameters():
            param.requires_grad = False

        if paired_mode:
            input_dim = 2 * embedding_dim + 1
        else:
            input_dim = embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2=None, distance_metric='cosine'):
        if self.paired_mode:
            if self.has_projection:
                emb1, emb2 = self.encoder(x1, x2, return_embedding=True)
            else:
                emb1, emb2 = self.encoder(x1, x2)

            if distance_metric == 'euclidean':
                distance = F.pairwise_distance(emb1, emb2, p=2)
            else:
                cosine_sim = torch.sum(emb1 * emb2, dim=1)
                distance = 1 - cosine_sim

            combined = torch.cat([emb1, emb2, distance.unsqueeze(1)], dim=1)
            logits = self.classifier(combined).squeeze()
        else:
            if self.has_projection:
                emb = self.encoder.forward_one(x1, return_embedding=True)
            else:
                emb = self.encoder.forward_one(x1)

            logits = self.classifier(emb).squeeze()

        return logits


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_probe(probe, train_loader, val_loader, device, epochs=10, lr=1e-3, weight_decay=1e-4,
                paired_mode=True, distance_metric='cosine'):
    """Train a probe with early stopping"""
    probe.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, probe.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Calculate class weights
    pos_count = sum(label.sum().item() for _, _, label, _, _ in train_loader)
    total_count = sum(len(label) for _, _, label, _, _ in train_loader)
    pos_weight = (total_count - pos_count) / pos_count if pos_count > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_ap = 0
    best_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        probe.train()
        for batch in train_loader:
            x1, x2, labels, _, _ = batch
            x1, x2, labels = x1.to(device), x2.to(device), labels.float().to(device)

            optimizer.zero_grad()

            if paired_mode:
                logits = probe(x1, x2, distance_metric=distance_metric)
            else:
                logits = probe(x1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        probe.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                x1, x2, labels, _, _ = batch
                x1, x2 = x1.to(device), x2.to(device)

                if paired_mode:
                    logits = probe(x1, x2, distance_metric=distance_metric)
                else:
                    logits = probe(x1)

                preds = torch.sigmoid(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_ap = average_precision_score(all_labels, all_preds)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = probe.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)

    return probe


def evaluate_probe_with_names(probe, dataloader, device, paired_mode=True, distance_metric='cosine'):
    """Evaluate probe and return predictions with names"""
    probe.eval()
    all_preds = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            x1, x2, labels, names, _ = batch
            x1, x2 = x1.to(device), x2.to(device)

            if paired_mode:
                logits = probe(x1, x2, distance_metric=distance_metric)
            else:
                logits = probe(x1)

            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_names.extend(names)

    return np.array(all_preds), np.array(all_labels), all_names


def evaluate_ce_model_with_names(model, dataloader, device):
    """Evaluate CE model and return predictions with names"""
    model.eval()
    all_preds = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            x, _, labels, names, _ = batch  # Unpack reconstructed format
            x = x.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs[:, 1]  # Positive class probability

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_names.extend(names)

    return np.array(all_preds), np.array(all_labels), all_names


# ============================================================================
# Metrics Functions
# ============================================================================

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
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred_binary)

    # Calibration
    cal_metrics = compute_calibration_metrics(y_true, y_pred_proba)
    metrics.update(cal_metrics)

    # Prediction statistics
    metrics['pred_mean'] = float(np.mean(y_pred_proba))
    metrics['pred_std'] = float(np.std(y_pred_proba))
    metrics['pos_ratio'] = float(np.mean(y_true))

    return metrics


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified evaluation framework for all model types")

    # Model type
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["paired", "reconstructed", "ce", "cross_attention"],
                       help="Type of model: paired (lung pairs), reconstructed (full CXR contrastive), ce (baseline), cross_attention (cross-attention siamese)")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="../split_node21_sets")
    parser.add_argument("--process", type=str, required=True,
                       choices=["lung_seg", "crop", "arch_seg"])
    parser.add_argument("--train_source", type=str, required=True,
                       choices=["chestxray14", "jsrt", "padchest"])

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to evaluate")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                       help="Name for checkpoint (defaults to filename)")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["rgb", "grey", "single", "rad", "randinit"])
    parser.add_argument("--embedding_dim", type=int, default=128,
                       help="Embedding dimension (for contrastive models)")
    parser.add_argument("--use_proj", action="store_true",
                       help="Set if checkpoint uses SiameseNetworkWithProjection")
    parser.add_argument("--no_single_embedding", action="store_true",
                       help="Set if checkpoint uses spatial models (SiameseNetworkSpatial)")
    parser.add_argument("--early_truncation", action="store_true",
                       help="Set if checkpoint uses early truncation models (SiameseNetworkSpatialEarly)")
    parser.add_argument("--num_attn_layers", type=int, default=1,
                       help="Number of cross-attention layers (for cross_attention models)")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads (for cross_attention models)")

    # Evaluation arguments
    parser.add_argument("--methods", type=str, nargs='+',
                       default=["distance", "linear", "shallow_mlp"],
                       choices=["distance", "linear", "shallow_mlp"],
                       help="Evaluation methods (ignored for CE models)")
    parser.add_argument("--distance_metric", type=str, default="cosine",
                       choices=["euclidean", "cosine"])
    parser.add_argument("--probe_epochs", type=int, default=10)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_wd", type=float, default=1e-4)

    # Other arguments
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--resize_dim", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path for summary metrics CSV")
    parser.add_argument("--predictions_csv", type=str, default=None,
                       help="Path for detailed predictions CSV (optional)")
    parser.add_argument("--cache_in_ram", action="store_true")
    parser.add_argument("--save_models", action="store_true",
                       help="Save trained probe models")
    parser.add_argument("--model_save_dir", type=str, default="evaluation_models")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build transforms
    base_transform = transforms.Compose([
        transforms.Resize((args.resize_dim, args.resize_dim)),
        transforms.ToTensor(),
    ])

    # Setup dataset paths
    external_test_names = ["chestxray14", "jsrt", "padchest"]
    external_test_names.remove(args.train_source)

    train_path = os.path.join(args.dataset_path, args.process, args.train_source, "train")
    test_path = os.path.join(args.dataset_path, args.process, args.train_source, "test")
    ext1_path = os.path.join(args.dataset_path, args.process, external_test_names[0], "test")
    ext2_path = os.path.join(args.dataset_path, args.process, external_test_names[1], "test")

    # Load datasets based on model type
    print("\nLoading datasets...")

    if args.model_type == "paired" or args.model_type == "cross_attention":
        # Paired lung models (including cross-attention)
        train_loader = helpers.dataloading.load_image_pair_dataset(
            dataset_path=train_path,
            batch_size=args.bsz,
            crop_size=args.resize_dim,
            symmetrical_transforms=True,
            class_to_idx={'nodule': 1, 'normal': 0},
            transform=base_transform,
            cache_in_ram=args.cache_in_ram,
            num_workers=args.workers
        )

        test_loader = helpers.dataloading.load_image_pair_dataset(
            dataset_path=test_path,
            batch_size=args.bsz,
            crop_size=args.resize_dim,
            symmetrical_transforms=True,
            class_to_idx={'nodule': 1, 'normal': 0},
            transform=base_transform,
            cache_in_ram=args.cache_in_ram,
            num_workers=args.workers
        )

        ext1_loader = helpers.dataloading.load_image_pair_dataset(
            dataset_path=ext1_path,
            batch_size=args.bsz,
            crop_size=args.resize_dim,
            symmetrical_transforms=True,
            class_to_idx={'nodule': 1, 'normal': 0},
            transform=base_transform,
            cache_in_ram=args.cache_in_ram,
            num_workers=args.workers
        )

        ext2_loader = helpers.dataloading.load_image_pair_dataset(
            dataset_path=ext2_path,
            batch_size=args.bsz,
            crop_size=args.resize_dim,
            symmetrical_transforms=True,
            class_to_idx={'nodule': 1, 'normal': 0},
            transform=base_transform,
            cache_in_ram=args.cache_in_ram,
            num_workers=args.workers
        )

    else:  # reconstructed or ce
        # Full CXR models (both use same dataset structure)
        train_base = helpers.dataloading.ImagePairDataset(
            root=train_path,
            transform=None,
            class_to_idx={'nodule': 1, 'normal': 0},
            cache_in_ram=args.cache_in_ram
        )
        train_dataset = ReconstructedCXRDataset(train_base, base_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.workers)

        test_base = helpers.dataloading.ImagePairDataset(
            root=test_path,
            transform=None,
            class_to_idx={'nodule': 1, 'normal': 0},
            cache_in_ram=args.cache_in_ram
        )
        test_dataset = ReconstructedCXRDataset(test_base, base_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.workers)

        ext1_base = helpers.dataloading.ImagePairDataset(
            root=ext1_path,
            transform=None,
            class_to_idx={'nodule': 1, 'normal': 0},
            cache_in_ram=args.cache_in_ram
        )
        ext1_dataset = ReconstructedCXRDataset(ext1_base, base_transform)
        ext1_loader = DataLoader(ext1_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.workers)

        ext2_base = helpers.dataloading.ImagePairDataset(
            root=ext2_path,
            transform=None,
            class_to_idx={'nodule': 1, 'normal': 0},
            cache_in_ram=args.cache_in_ram
        )
        ext2_dataset = ReconstructedCXRDataset(ext2_base, base_transform)
        ext2_loader = DataLoader(ext2_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.workers)

    dataloaders = {
        'train': train_loader,
        'test': test_loader,
        external_test_names[0]: ext1_loader,
        external_test_names[1]: ext2_loader
    }

    all_results = []
    all_predictions = []

    # Get checkpoint name
    checkpoint_name = args.checkpoint_name
    if checkpoint_name is None:
        checkpoint_name = Path(args.checkpoint).stem

    # Create model save directory if needed
    if args.save_models:
        save_dir = Path(args.model_save_dir) / checkpoint_name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Models will be saved to: {save_dir}")

    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"{'='*80}")

    # Load model based on type
    if args.model_type == "ce":
        # ====================================================================
        # CE Baseline Model
        # ====================================================================
        model = helpers.models.load_full_model(
            model_name=args.model_name,
            num_classes=2,
            freeze_backbone=False,
            device=device
        )
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Loaded CE model from epoch {checkpoint.get('epoch', 'unknown')}")

        # Evaluate CE model
        for ds_name, dataloader in dataloaders.items():
            print(f"\nEvaluating on {ds_name}...")
            probs, labels, names = evaluate_ce_model_with_names(model, dataloader, device)
            metrics = evaluate_predictions(labels, probs)

            result = {
                'checkpoint': checkpoint_name,
                'method': 'ce_baseline',
                'dataset': ds_name,
                **metrics
            }
            all_results.append(result)

            # Save individual predictions if requested
            if args.predictions_csv:
                threshold = metrics['threshold']
                predictions = (probs >= threshold).astype(int)
                correct = (predictions == labels).astype(int)

                for i, name in enumerate(names):
                    pred_row = {
                        'checkpoint': checkpoint_name,
                        'method': 'ce_baseline',
                        'dataset': ds_name,
                        'sample_name': name,
                        'true_label': int(labels[i]),
                        'predicted_prob': float(probs[i]),
                        'predicted_label': int(predictions[i]),
                        'correct': int(correct[i]),
                        'threshold': float(threshold)
                    }
                    all_predictions.append(pred_row)

            print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

    elif args.model_type == "cross_attention":
        # ====================================================================
        # Cross-Attention Siamese Model
        # ====================================================================
        # Load backbone without pooling (need spatial features for cross-attention)
        backbone = helpers.models.load_truncated_model(
            args.model_name,
            single_embedding=False,  # Need spatial features (7x7)
            truncation_layer=4
        )

        # Create cross-attention model
        encoder = helpers.crossattention.CrossAttentionSiamese(
            backbone,
            embedding_dim=args.embedding_dim,
            num_attn_layers=args.num_attn_layers,
            num_heads=args.num_heads,
            freeze_backbone=True
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        encoder.eval()

        print(f"Loaded cross-attention model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Attention layers: {args.num_attn_layers}, Heads: {args.num_heads}")

        paired_mode = True

        # Evaluate with each method
        for method in args.methods:
            print(f"\n--- Method: {method} ---")

            if method == "distance":
                # Distance-based classification
                classifier = DistanceClassifierPaired(encoder, device)
                classifier.fit(train_loader)

                for ds_name, dataloader in dataloaders.items():
                    probs, labels, names = classifier.predict_proba_with_names(dataloader)
                    metrics = evaluate_predictions(labels, probs)

                    result = {
                        'checkpoint': checkpoint_name,
                        'method': 'distance',
                        'dataset': ds_name,
                        **metrics
                    }
                    all_results.append(result)

                    # Save individual predictions if requested
                    if args.predictions_csv:
                        threshold = metrics['threshold']
                        predictions = (probs >= threshold).astype(int)
                        correct = (predictions == labels).astype(int)

                        for i, name in enumerate(names):
                            pred_row = {
                                'checkpoint': checkpoint_name,
                                'method': 'distance',
                                'dataset': ds_name,
                                'sample_name': name,
                                'true_label': int(labels[i]),
                                'predicted_prob': float(probs[i]),
                                'predicted_label': int(predictions[i]),
                                'correct': int(correct[i]),
                                'threshold': float(threshold)
                            }
                            all_predictions.append(pred_row)

                    print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

                # Save distance classifier
                if args.save_models:
                    distance_info = {
                        'optimal_threshold': classifier.optimal_threshold,
                        'method': 'distance'
                    }
                    torch.save(distance_info, save_dir / 'distance_classifier.pth')
                    print(f"  Saved distance classifier to {save_dir / 'distance_classifier.pth'}")

            elif method == "linear":
                # Linear probe
                probe = LinearProbe(encoder, args.embedding_dim, paired_mode=paired_mode)
                probe = train_probe(
                    probe, train_loader, test_loader, device,
                    epochs=args.probe_epochs,
                    lr=args.probe_lr,
                    weight_decay=args.probe_wd,
                    paired_mode=paired_mode,
                    distance_metric=args.distance_metric
                )

                for ds_name, dataloader in dataloaders.items():
                    probs, labels, names = evaluate_probe_with_names(
                        probe, dataloader, device,
                        paired_mode=paired_mode,
                        distance_metric=args.distance_metric
                    )
                    metrics = evaluate_predictions(labels, probs)

                    result = {
                        'checkpoint': checkpoint_name,
                        'method': 'linear_probe',
                        'dataset': ds_name,
                        **metrics
                    }
                    all_results.append(result)

                    # Save individual predictions if requested
                    if args.predictions_csv:
                        threshold = metrics['threshold']
                        predictions = (probs >= threshold).astype(int)
                        correct = (predictions == labels).astype(int)

                        for i, name in enumerate(names):
                            pred_row = {
                                'checkpoint': checkpoint_name,
                                'method': 'linear_probe',
                                'dataset': ds_name,
                                'sample_name': name,
                                'true_label': int(labels[i]),
                                'predicted_prob': float(probs[i]),
                                'predicted_label': int(predictions[i]),
                                'correct': int(correct[i]),
                                'threshold': float(threshold)
                            }
                            all_predictions.append(pred_row)

                    print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

                # Save linear probe
                if args.save_models:
                    torch.save(probe.state_dict(), save_dir / 'linear_probe.pth')
                    print(f"  Saved linear probe to {save_dir / 'linear_probe.pth'}")

            elif method == "shallow_mlp":
                # Shallow MLP probe
                probe = ShallowMLPProbe(encoder, args.embedding_dim, paired_mode=paired_mode)
                probe = train_probe(
                    probe, train_loader, test_loader, device,
                    epochs=args.probe_epochs,
                    lr=args.probe_lr,
                    weight_decay=args.probe_wd,
                    paired_mode=paired_mode,
                    distance_metric=args.distance_metric
                )

                for ds_name, dataloader in dataloaders.items():
                    probs, labels, names = evaluate_probe_with_names(
                        probe, dataloader, device,
                        paired_mode=paired_mode,
                        distance_metric=args.distance_metric
                    )
                    metrics = evaluate_predictions(labels, probs)

                    result = {
                        'checkpoint': checkpoint_name,
                        'method': 'shallow_mlp_probe',
                        'dataset': ds_name,
                        **metrics
                    }
                    all_results.append(result)

                    # Save individual predictions if requested
                    if args.predictions_csv:
                        threshold = metrics['threshold']
                        predictions = (probs >= threshold).astype(int)
                        correct = (predictions == labels).astype(int)

                        for i, name in enumerate(names):
                            pred_row = {
                                'checkpoint': checkpoint_name,
                                'method': 'shallow_mlp_probe',
                                'dataset': ds_name,
                                'sample_name': name,
                                'true_label': int(labels[i]),
                                'predicted_prob': float(probs[i]),
                                'predicted_label': int(predictions[i]),
                                'correct': int(correct[i]),
                                'threshold': float(threshold)
                            }
                            all_predictions.append(pred_row)

                    print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

                # Save shallow MLP probe
                if args.save_models:
                    torch.save(probe.state_dict(), save_dir / 'shallow_mlp_probe.pth')
                    print(f"  Saved shallow MLP probe to {save_dir / 'shallow_mlp_probe.pth'}")

    else:
        # ====================================================================
        # Contrastive Models (Paired or Reconstructed)
        # ====================================================================
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

        # Select model architecture based on spatial flags
        if args.no_single_embedding and args.early_truncation:
            # Use early spatial models (layer3, 14x14 resolution)
            if args.use_proj:
                encoder = helpers.models.SiameseNetworkWithProjectionSpatialEarly(
                    backbone,
                    embedding_dim=args.embedding_dim,
                    projection_dim=128,
                    freeze_backbone=True
                ).to(device)
            else:
                encoder = helpers.models.SiameseNetworkSpatialEarly(
                    backbone,
                    embedding_dim=args.embedding_dim,
                    freeze_backbone=True
                ).to(device)
        elif args.no_single_embedding:
            # Use spatial models that process 7x7 feature maps with 1x1 convs
            if args.use_proj:
                encoder = helpers.models.SiameseNetworkWithProjectionSpatial(
                    backbone,
                    embedding_dim=args.embedding_dim,
                    projection_dim=128,
                    freeze_backbone=True
                ).to(device)
            else:
                encoder = helpers.models.SiameseNetworkSpatial(
                    backbone,
                    embedding_dim=args.embedding_dim,
                    freeze_backbone=True
                ).to(device)
        else:
            # Use standard models that pool immediately to 2048-dim vectors
            if args.use_proj:
                encoder = helpers.models.SiameseNetworkWithProjection(
                    backbone,
                    embedding_dim=args.embedding_dim,
                    projection_dim=128,
                    freeze_backbone=True
                ).to(device)
            else:
                encoder = helpers.models.SiameseNetwork(
                    backbone,
                    embedding_dim=args.embedding_dim,
                    freeze_backbone=True
                ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        encoder.eval()

        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        paired_mode = (args.model_type == "paired")

        # Evaluate with each method
        for method in args.methods:
            print(f"\n--- Method: {method} ---")

            if method == "distance":
                # Distance-based classification
                if paired_mode:
                    classifier = DistanceClassifierPaired(encoder, device)
                else:
                    classifier = DistanceClassifierFullImage(encoder, device)

                classifier.fit(train_loader)

                for ds_name, dataloader in dataloaders.items():
                    probs, labels, names = classifier.predict_proba_with_names(dataloader)
                    metrics = evaluate_predictions(labels, probs)

                    result = {
                        'checkpoint': checkpoint_name,
                        'method': 'distance',
                        'dataset': ds_name,
                        **metrics
                    }
                    all_results.append(result)

                    # Save individual predictions if requested
                    if args.predictions_csv:
                        threshold = metrics['threshold']
                        predictions = (probs >= threshold).astype(int)
                        correct = (predictions == labels).astype(int)

                        for i, name in enumerate(names):
                            pred_row = {
                                'checkpoint': checkpoint_name,
                                'method': 'distance',
                                'dataset': ds_name,
                                'sample_name': name,
                                'true_label': int(labels[i]),
                                'predicted_prob': float(probs[i]),
                                'predicted_label': int(predictions[i]),
                                'correct': int(correct[i]),
                                'threshold': float(threshold)
                            }
                            all_predictions.append(pred_row)

                    print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

                # Save distance classifier
                if args.save_models:
                    if paired_mode:
                        distance_info = {
                            'optimal_threshold': classifier.optimal_threshold,
                            'method': 'distance'
                        }
                    else:
                        distance_info = {
                            'optimal_threshold': classifier.optimal_threshold,
                            'class_prototypes': classifier.class_prototypes,
                            'method': 'distance'
                        }
                    torch.save(distance_info, save_dir / 'distance_classifier.pth')
                    print(f"  Saved distance classifier to {save_dir / 'distance_classifier.pth'}")

            elif method == "linear":
                # Linear probe
                probe = LinearProbe(encoder, args.embedding_dim, paired_mode=paired_mode)
                probe = train_probe(
                    probe, train_loader, test_loader, device,
                    epochs=args.probe_epochs,
                    lr=args.probe_lr,
                    weight_decay=args.probe_wd,
                    paired_mode=paired_mode,
                    distance_metric=args.distance_metric
                )

                for ds_name, dataloader in dataloaders.items():
                    probs, labels, names = evaluate_probe_with_names(
                        probe, dataloader, device,
                        paired_mode=paired_mode,
                        distance_metric=args.distance_metric
                    )
                    metrics = evaluate_predictions(labels, probs)

                    result = {
                        'checkpoint': checkpoint_name,
                        'method': 'linear_probe',
                        'dataset': ds_name,
                        **metrics
                    }
                    all_results.append(result)

                    # Save individual predictions if requested
                    if args.predictions_csv:
                        threshold = metrics['threshold']
                        predictions = (probs >= threshold).astype(int)
                        correct = (predictions == labels).astype(int)

                        for i, name in enumerate(names):
                            pred_row = {
                                'checkpoint': checkpoint_name,
                                'method': 'linear_probe',
                                'dataset': ds_name,
                                'sample_name': name,
                                'true_label': int(labels[i]),
                                'predicted_prob': float(probs[i]),
                                'predicted_label': int(predictions[i]),
                                'correct': int(correct[i]),
                                'threshold': float(threshold)
                            }
                            all_predictions.append(pred_row)

                    print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

                # Save linear probe
                if args.save_models:
                    torch.save({
                        'model_state_dict': probe.state_dict(),
                        'embedding_dim': args.embedding_dim,
                        'method': 'linear_probe'
                    }, save_dir / 'linear_probe.pth')
                    print(f"  Saved linear probe to {save_dir / 'linear_probe.pth'}")

            elif method == "shallow_mlp":
                # Shallow MLP probe
                probe = ShallowMLPProbe(encoder, args.embedding_dim, hidden_dim=64, dropout=0.3, paired_mode=paired_mode)
                probe = train_probe(
                    probe, train_loader, test_loader, device,
                    epochs=args.probe_epochs,
                    lr=args.probe_lr,
                    weight_decay=args.probe_wd,
                    paired_mode=paired_mode,
                    distance_metric=args.distance_metric
                )

                for ds_name, dataloader in dataloaders.items():
                    probs, labels, names = evaluate_probe_with_names(
                        probe, dataloader, device,
                        paired_mode=paired_mode,
                        distance_metric=args.distance_metric
                    )
                    metrics = evaluate_predictions(labels, probs)

                    result = {
                        'checkpoint': checkpoint_name,
                        'method': 'shallow_mlp',
                        'dataset': ds_name,
                        **metrics
                    }
                    all_results.append(result)

                    # Save individual predictions if requested
                    if args.predictions_csv:
                        threshold = metrics['threshold']
                        predictions = (probs >= threshold).astype(int)
                        correct = (predictions == labels).astype(int)

                        for i, name in enumerate(names):
                            pred_row = {
                                'checkpoint': checkpoint_name,
                                'method': 'shallow_mlp',
                                'dataset': ds_name,
                                'sample_name': name,
                                'true_label': int(labels[i]),
                                'predicted_prob': float(probs[i]),
                                'predicted_label': int(predictions[i]),
                                'correct': int(correct[i]),
                                'threshold': float(threshold)
                            }
                            all_predictions.append(pred_row)

                    print(f"  {ds_name:12s} - AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

                # Save shallow MLP probe
                if args.save_models:
                    torch.save({
                        'model_state_dict': probe.state_dict(),
                        'embedding_dim': args.embedding_dim,
                        'hidden_dim': 64,
                        'dropout': 0.3,
                        'method': 'shallow_mlp'
                    }, save_dir / 'shallow_mlp_probe.pth')
                    print(f"  Saved shallow MLP probe to {save_dir / 'shallow_mlp_probe.pth'}")

    # Save results
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(args.output_csv, index=False)

    print(f"\n{'='*80}")
    print(f"Summary metrics saved to: {args.output_csv}")

    if args.predictions_csv:
        df_predictions = pd.DataFrame(all_predictions)
        df_predictions.to_csv(args.predictions_csv, index=False)
        print(f"Detailed predictions saved to: {args.predictions_csv}")

    print(f"{'='*80}")

    # Print summary
    print("\nSummary (Test set):")
    summary = df_summary[df_summary['dataset'] == 'test'][['checkpoint', 'method', 'ap', 'auc', 'f1']]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
