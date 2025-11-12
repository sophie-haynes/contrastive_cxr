#!/usr/bin/env python3
"""
Joint Hierarchical + CE training with spatial feature concatenation.

This script trains a Siamese network with TWO objectives:
1. Hierarchical loss: Pair contrastive + SupCon on individual lungs
2. CE loss: Classify pairs using spatially concatenated features

Architecture:
- Each lung → Backbone → Spatial features (B, 128, 7, 7)
- Hierarchical: Uses individual lung embeddings (after pooling)
  - Pair contrastive: Push nodule pairs apart from normal pairs
  - SupCon: Cluster individual lungs by their labels
- CE: Uses concatenated spatial features (before pooling)
- Both losses share the same backbone → joint gradients

This combines the best of both worlds:
- Hierarchical loss exploits pair structure at multiple levels
- CE uses reconstructed feature representation for classification
"""
import os
import sys
import csv
import argparse

sys.path.insert(1, '../')
import helpers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# Import the Hierarchical loss
from helpers.supcon import HierarchicalLoss

# Adding seed stuff
import random
import numpy as np

def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class JointHierarchicalCEModel(nn.Module):
    """
    Model that supports both Hierarchical and CE objectives.

    - Hierarchical: Uses individual lung embeddings (after pooling)
    - CE: Uses spatially concatenated features (before pooling)
    """
    def __init__(self, siamese_model, num_classes=2):
        super(JointHierarchicalCEModel, self).__init__()
        self.siamese_model = siamese_model

        # Classification head operates on concatenated features
        self.classifier = nn.Linear(128, num_classes)

    def forward_hierarchical(self, x1, x2):
        """
        Forward pass for Hierarchical loss.

        Returns pooled embeddings for each lung separately.
        """
        # Get embeddings (after pooling) for hierarchical loss
        emb1, emb2 = self.siamese_model(x1, x2, return_logits=False)
        return emb1, emb2

    def forward_ce(self, x1, x2):
        """
        Forward pass for CE loss.

        Returns logits from spatially concatenated features.
        """
        # Get spatial features before pooling
        spatial1 = self.siamese_model.forward_one_spatial(x1)  # (B, 128, 7, 7)
        spatial2 = self.siamese_model.forward_one_spatial(x2)  # (B, 128, 7, 7)

        # Concatenate horizontally to mimic reconstruction
        concatenated = torch.cat([spatial1, spatial2], dim=3)  # (B, 128, 7, 14)

        # Global average pooling
        pooled = F.adaptive_avg_pool2d(concatenated, output_size=1)  # (B, 128, 1, 1)
        features = torch.flatten(pooled, 1)  # (B, 128)

        # Classify
        logits = self.classifier(features)
        return logits

    def forward(self, x1, x2):
        """
        Full forward pass returning both Hierarchical embeddings and CE logits.

        Returns:
            emb1, emb2: Individual lung embeddings for Hierarchical loss
            logits: Classification logits for CE
        """
        emb1, emb2 = self.forward_hierarchical(x1, x2)
        logits = self.forward_ce(x1, x2)
        return emb1, emb2, logits


def save_model_checkpoint(model, optimizer, epoch, train_loss, eval_metrics, args,
                         save_dir='checkpoints', is_best=False, is_best_accuracy=False):
    """Save model checkpoint with organized structure"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_metrics': eval_metrics,
        'args': vars(args)
    }

    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model (silhouette) saved at epoch {epoch}!")

    if is_best_accuracy:
        best_acc_path = os.path.join(save_dir, 'best_model_accuracy.pth')
        torch.save(checkpoint, best_acc_path)
        print(f"New best model (accuracy) saved at epoch {epoch}!")

    return checkpoint_path


def build_transforms(resize_dim: int, single=False, aggressive_aug=False):
    """Build transforms for training and testing"""
    if not single:
        base_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])

        if aggressive_aug:
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=resize_dim, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.4, hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])
        else:
            augment_transform = transforms.Compose([
                transforms.Resize((resize_dim, resize_dim)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
    else:
        base_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])

        if aggressive_aug:
            augment_transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.RandomResizedCrop(size=resize_dim, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4)
                ], p=0.8),
                transforms.ToTensor(),
            ])
        else:
            augment_transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((resize_dim, resize_dim)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
    return base_transform, augment_transform


def train_joint_epoch(model, dataloader, hierarchical_criterion, ce_criterion, optimizer,
                      device, ce_weight=1.0):
    """
    Training loop for joint Hierarchical + CE

    Args:
        model: JointHierarchicalCEModel
        dataloader: DataLoader with individual lung labels
        hierarchical_criterion: HierarchicalLoss
        ce_criterion: CrossEntropyLoss
        optimizer: Optimizer
        device: Device
        ce_weight: Weight for CE loss (Hierarchical weight is 1.0)
    """
    model.train()
    total_loss = 0.0
    total_hierarchical_loss = 0.0
    total_pair_loss = 0.0
    total_supcon_loss = 0.0
    total_ce_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        # Unpack batch
        img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
        img1, img2 = img1.to(device), img2.to(device)
        label1, label2 = label1.to(device), label2.to(device)
        pair_labels = pair_class_idx.to(device)

        optimizer.zero_grad()

        # Forward pass
        emb1, emb2, logits = model(img1, img2)

        # 1. Hierarchical loss (pair contrastive + SupCon)
        hierarchical_loss, pair_loss, supcon_loss = hierarchical_criterion(
            emb1, emb2, pair_labels, label1, label2
        )

        # 2. CE loss on pairs (using concatenated features)
        ce_loss = ce_criterion(logits, pair_labels)

        # 3. Combine losses
        loss = hierarchical_loss + ce_weight * ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_hierarchical_loss += hierarchical_loss.item()
        total_pair_loss += pair_loss.item()
        total_supcon_loss += supcon_loss.item()
        total_ce_loss += ce_loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += pair_labels.size(0)
        correct += (predicted == pair_labels).sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_hierarchical_loss = total_hierarchical_loss / max(len(dataloader), 1)
    avg_pair_loss = total_pair_loss / max(len(dataloader), 1)
    avg_supcon_loss = total_supcon_loss / max(len(dataloader), 1)
    avg_ce_loss = total_ce_loss / max(len(dataloader), 1)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, avg_hierarchical_loss, avg_pair_loss, avg_supcon_loss, avg_ce_loss, accuracy


def eval_joint_epoch(model, dataloader, hierarchical_criterion, ce_criterion, device, ce_weight=1.0):
    """
    Evaluation loop with comprehensive metrics

    Metrics tracked:
    - Joint loss (Hierarchical + CE)
    - Hierarchical loss component (pair + supcon)
    - Pair loss component
    - SupCon loss component
    - CE loss component
    - Classification accuracy (pair-level)
    - Silhouette score (individual lung clustering)
    - Davies-Bouldin score
    - Embedding std (collapse detection)
    - Pair distances
    """
    model.eval()
    total_loss = 0.0
    total_hierarchical_loss = 0.0
    total_pair_loss = 0.0
    total_supcon_loss = 0.0
    total_ce_loss = 0.0
    correct = 0
    total = 0

    # Track pair distances
    normal_pair_distances = []
    nodule_pair_distances = []

    # Collect individual lung embeddings for quality metrics
    all_embeddings_list = []
    all_individual_labels_list = []

    # Collect concatenated features for pair-level metrics
    all_pair_features_list = []
    all_pair_labels_list = []

    import numpy as np
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
            img1, img2 = img1.to(device), img2.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            pair_labels = pair_class_idx.to(device)

            # Forward pass
            emb1, emb2, logits = model(img1, img2)

            # Store individual embeddings and labels
            all_embeddings_list.append(emb1.cpu())
            all_embeddings_list.append(emb2.cpu())
            all_individual_labels_list.append(label1.cpu())
            all_individual_labels_list.append(label2.cpu())

            # Compute Hierarchical loss
            hierarchical_loss, pair_loss, supcon_loss = hierarchical_criterion(
                emb1, emb2, pair_labels, label1, label2
            )

            # Compute CE loss
            ce_loss = ce_criterion(logits, pair_labels)

            # Combined loss
            loss = hierarchical_loss + ce_weight * ce_loss

            total_loss += loss.item()
            total_hierarchical_loss += hierarchical_loss.item()
            total_pair_loss += pair_loss.item()
            total_supcon_loss += supcon_loss.item()
            total_ce_loss += ce_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += pair_labels.size(0)
            correct += (predicted == pair_labels).sum().item()

            # Calculate distances between pairs
            distances = F.pairwise_distance(emb1, emb2, p=2)
            normal_pair_mask = (pair_labels == 0)
            nodule_pair_mask = (pair_labels == 1)

            if normal_pair_mask.sum() > 0:
                normal_pair_distances.extend(distances[normal_pair_mask].cpu().numpy())
            if nodule_pair_mask.sum() > 0:
                nodule_pair_distances.extend(distances[nodule_pair_mask].cpu().numpy())

            # Get concatenated features for pair-level silhouette
            spatial1 = model.siamese_model.forward_one_spatial(img1)
            spatial2 = model.siamese_model.forward_one_spatial(img2)
            concatenated = torch.cat([spatial1, spatial2], dim=3)
            pooled = F.adaptive_avg_pool2d(concatenated, output_size=1)
            pair_features = torch.flatten(pooled, 1)

            all_pair_features_list.append(pair_features.cpu())
            all_pair_labels_list.append(pair_labels.cpu())

    # Concatenate all individual embeddings
    all_embeddings_np = torch.cat(all_embeddings_list, dim=0).numpy()
    all_individual_labels_np = torch.cat(all_individual_labels_list, dim=0).numpy()

    # Concatenate all pair features
    all_pair_features_np = torch.cat(all_pair_features_list, dim=0).numpy()
    all_pair_labels_np = torch.cat(all_pair_labels_list, dim=0).numpy()

    # Calculate basic metrics
    avg_loss = total_loss / max(len(dataloader), 1)
    avg_hierarchical_loss = total_hierarchical_loss / max(len(dataloader), 1)
    avg_pair_loss = total_pair_loss / max(len(dataloader), 1)
    avg_supcon_loss = total_supcon_loss / max(len(dataloader), 1)
    avg_ce_loss = total_ce_loss / max(len(dataloader), 1)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Pair distance metrics
    normal_pair_mean = np.mean(normal_pair_distances) if normal_pair_distances else 0
    nodule_pair_mean = np.mean(nodule_pair_distances) if nodule_pair_distances else 0
    separation = nodule_pair_mean - normal_pair_mean

    # Calculate embedding quality metrics using INDIVIDUAL labels (for Hierarchical)
    silhouette_individual = silhouette_score(all_embeddings_np, all_individual_labels_np) if len(np.unique(all_individual_labels_np)) > 1 else 0
    davies_bouldin_individual = davies_bouldin_score(all_embeddings_np, all_individual_labels_np) if len(np.unique(all_individual_labels_np)) > 1 else 0

    # Calculate pair-level silhouette (for CE)
    silhouette_pair = silhouette_score(all_pair_features_np, all_pair_labels_np) if len(np.unique(all_pair_labels_np)) > 1 else 0

    # Embedding std (collapse detection)
    embedding_std = np.std(all_embeddings_np)

    # Count samples
    num_normal_lungs = np.sum(all_individual_labels_np == 0)
    num_nodule_lungs = np.sum(all_individual_labels_np == 1)
    num_normal_pairs = np.sum(all_pair_labels_np == 0)
    num_nodule_pairs = np.sum(all_pair_labels_np == 1)

    metrics = {
        'avg_loss': avg_loss,
        'hierarchical_loss': avg_hierarchical_loss,
        'pair_loss': avg_pair_loss,
        'supcon_loss': avg_supcon_loss,
        'ce_loss': avg_ce_loss,
        'accuracy': accuracy,
        'normal_pair_dist_mean': normal_pair_mean,
        'nodule_pair_dist_mean': nodule_pair_mean,
        'separation': separation,
        'silhouette_individual': silhouette_individual,
        'silhouette_pair': silhouette_pair,
        'davies_bouldin_individual': davies_bouldin_individual,
        'embedding_std': embedding_std,
        'num_normal_lungs': int(num_normal_lungs),
        'num_nodule_lungs': int(num_nodule_lungs),
        'num_normal_pairs': int(num_normal_pairs),
        'num_nodule_pairs': int(num_nodule_pairs)
    }

    return metrics


def eval_joint_epoch_with_viz(model, dataloader, hierarchical_criterion, ce_criterion, device,
                               epoch, testset, plt_path="plots/hierarchical_ce_spatial_concat",
                               ce_weight=1.0):
    """Evaluation with visualization"""
    metrics = eval_joint_epoch(model, dataloader, hierarchical_criterion, ce_criterion, device, ce_weight)

    import numpy as np

    # Collect individual lung embeddings for visualization
    all_embeddings_list = []
    all_individual_labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
            img1, img2 = img1.to(device), img2.to(device)
            label1, label2 = label1.to(device), label2.to(device)

            # Get individual embeddings
            emb1, emb2 = model.forward_hierarchical(img1, img2)

            all_embeddings_list.append(emb1.cpu())
            all_embeddings_list.append(emb2.cpu())
            all_individual_labels_list.append(label1.cpu())
            all_individual_labels_list.append(label2.cpu())

    # Concatenate
    all_embeddings = torch.cat(all_embeddings_list, dim=0).numpy()
    all_labels = torch.cat(all_individual_labels_list, dim=0).numpy()

    # t-SNE plot with individual labels
    helpers.viz.plot_tsne_from_embeddings(
        all_embeddings,
        all_labels,
        epoch=epoch,
        save_path=os.path.join(plt_path, f'{testset}_tsne_hierarchical_ce_epoch_{epoch}.png')
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with joint Hierarchical + CE using spatial feature concatenation"
    )
    parser.add_argument("--model_name", type=str,
                       choices=["rgb", "grey", "single", "rad", "randinit"],
                       required=True)
    parser.add_argument("--resize_dim", type=int, default=224)
    parser.add_argument("--dataset_path", type=str, default="../split_node21_sets")
    parser.add_argument("--label_csv", type=str, default="../split_node21_sets/only_nodule_half_labels.csv",
                       help="Path to CSV with individual lung labels")
    parser.add_argument("--process", type=str,
                       choices=["lung_seg", "crop", "arch_seg"], required=True)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--cache_in_ram", action="store_true")
    parser.add_argument("--symmetrical_transforms", action="store_true")
    parser.add_argument("--aggressive_aug", action="store_true",
                       help="Use official SupContrast augmentations (stronger)")
    parser.add_argument("--freeze_backbone", action="store_true")

    # Hierarchical loss configuration
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for pair contrastive loss in hierarchical")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="Weight for SupCon loss in hierarchical")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin for pair contrastive loss")
    parser.add_argument("--distance", type=str, default="euclidean",
                       choices=["euclidean", "cosine"],
                       help="Distance metric for pair loss")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for SupCon loss")

    # CE weight
    parser.add_argument("--ce_weight", type=float, default=1.0,
                       help="Weight for CE loss (Hierarchical weight is 1.0)")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plot_freq", type=int, default=5)
    parser.add_argument("--plot_dir", type=str, default="plots/hierarchical_ce_spatial_concat")
    parser.add_argument("--log_dir", type=str, default="logs/hierarchical_ce_spatial_concat")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument('-q', '--quiet', action="store_true")
    parser.add_argument("--train_source", type=str,
                       choices=["chestxray14", "jsrt", "padchest"],
                       required=True)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--no_tsne', action="store_true")
    parser.add_argument('--no_single_embedding', action="store_true",
                       help="Use 4D backbone output (B, 2048, 7, 7) - REQUIRED")
    parser.add_argument('--early_truncation', action="store_true",
                       help="Use layer3 truncation (B, 1024, 14, 14). Requires --no_single_embedding")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None,
                       help="Specific epoch to resume from")

    return parser.parse_args()


def main():
    args = parse_args()

    # Spatial concat requires spatial models
    if not args.no_single_embedding:
        raise ValueError("--no_single_embedding is REQUIRED for spatial concatenation")

    # set seed for reproducibility
    set_seed(args.seed)

    # Transforms
    base_transform, augment_transform = build_transforms(
        args.resize_dim,
        single=(args.model_name == "single"),
        aggressive_aug=args.aggressive_aug
    )

    # Setup paths
    external_test_names = ["chestxray14", "jsrt", "padchest"]
    train_path = os.path.join(args.dataset_path, args.process, args.train_source, "train")
    test_path = os.path.join(args.dataset_path, args.process, args.train_source, "test")

    external_test_names.remove(args.train_source)
    test2_path = os.path.join(args.dataset_path, args.process, external_test_names[0], "test")
    test3_path = os.path.join(args.dataset_path, args.process, external_test_names[1], "test")

    # Dataloaders with individual labels
    print("Loading training data with individual lung labels...")
    train_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=train_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=augment_transform,
        cache_in_ram=args.cache_in_ram,
        single=(args.model_name == "single"),
        num_workers=args.workers
    )

    print("\nLoading test data with individual lung labels...")
    test_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=test_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=base_transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )

    print("\nLoading external test data 1...")
    test2_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=test2_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=base_transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )

    print("\nLoading external test data 2...")
    test3_dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=test3_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        pair_class_to_idx={'nodule': 1, 'normal': 0},
        lung_class_to_idx={'normal': 0, 'nodule': 1},
        transform=base_transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    if args.early_truncation and not args.no_single_embedding:
        raise ValueError("--early_truncation requires --no_single_embedding flag")

    # Load backbone
    truncation_layer = 3 if args.early_truncation else 4
    backbone = helpers.models.load_truncated_model(
        args.model_name,
        single_embedding=not args.no_single_embedding,
        truncation_layer=truncation_layer
    )

    # Create Siamese model (no classification head)
    if args.early_truncation:
        siamese_model = helpers.models.SiameseNetworkSpatialEarly(
            backbone,
            embedding_dim=128,
            freeze_backbone=args.freeze_backbone,
            num_classes=None
        ).to(device)
    else:
        siamese_model = helpers.models.SiameseNetworkSpatial(
            backbone,
            embedding_dim=128,
            freeze_backbone=args.freeze_backbone,
            num_classes=None
        ).to(device)

    # Wrap with joint model
    model = JointHierarchicalCEModel(siamese_model, num_classes=2).to(device)

    # Loss functions
    hierarchical_criterion = HierarchicalLoss(
        alpha=args.alpha,
        beta=args.beta,
        margin=args.margin,
        distance=args.distance,
        temperature=args.temperature
    )
    ce_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup logging directories
    exp_name_parts = [
        args.model_name,
        "hierarchical_ce_concat",  # Identifier
        "frz" if args.freeze_backbone else "unfrz",
        args.process,
        "sym" if args.symmetrical_transforms else "nosym",
        f"bsz{args.bsz}",
        f"lr{args.lr}",
        f"alpha{args.alpha}_beta{args.beta}",
        f"temp{args.temperature}",
        f"cew{args.ce_weight}",
        f"seed{args.seed}",
        str(args.run)
    ]

    if args.early_truncation:
        exp_name_parts.insert(1, "early")

    if args.aggressive_aug:
        exp_name_parts.insert(1, "aggaug")

    exp_name = "_".join(exp_name_parts)

    # Handle checkpoint resumption
    start_epoch = 0
    if args.resume or args.resume_epoch is not None:
        if args.resume:
            checkpoint_path = args.resume
        else:
            log_dir_temp = os.path.join(args.log_dir, args.train_source, exp_name)
            checkpoint_path = os.path.join(log_dir_temp, "checkpoints", f"checkpoint_epoch_{args.resume_epoch}.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    if args.comment != "":
        exp_name = "_".join([exp_name, args.comment])

    plot_dir = os.path.join(args.plot_dir, args.train_source, exp_name)
    log_dir = os.path.join(args.log_dir, args.train_source, exp_name)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Log configuration
    config_str = (f"Training with Joint Hierarchical + CE (Spatial Concatenation):\n"
                 f"  model_name={args.model_name}, resize_dim={args.resize_dim}\n"
                 f"  alpha={args.alpha}, beta={args.beta}\n"
                 f"  margin={args.margin}, distance={args.distance}\n"
                 f"  temperature={args.temperature}, ce_weight={args.ce_weight}\n"
                 f"  bsz={args.bsz}, lr={args.lr}, epochs={args.epochs}\n"
                 f"  early_truncation={args.early_truncation}\n"
                 f"  aggressive_aug={args.aggressive_aug}\n"
                 f"  seed={args.seed}, comment={args.comment}\n"
                 f"  device={device}")

    print("\n" + "="*60)
    print(config_str)
    print("="*60 + "\n")

    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(config_str)

    # Setup CSV logging
    csv_cols = ['avg_loss', 'hierarchical_loss', 'pair_loss', 'supcon_loss', 'ce_loss', 'accuracy',
                'normal_pair_dist_mean', 'nodule_pair_dist_mean', 'separation',
                'silhouette_individual', 'silhouette_pair', 'davies_bouldin_individual',
                'embedding_std', 'num_normal_lungs', 'num_nodule_lungs',
                'num_normal_pairs', 'num_nodule_pairs']

    if start_epoch == 0:
        for dataset_name in [f"train-{args.train_source}",
                            f"test-{args.train_source}",
                            f"test-{external_test_names[0]}",
                            f"test-{external_test_names[1]}"]:
            with open(os.path.join(log_dir, f"{dataset_name}_results.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_cols)
                writer.writeheader()
    else:
        print(f"Resuming: Will append to existing CSV files in {log_dir}")

    # Training loop
    if args.quiet:
        epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress")

    best_silhouette = -float('inf')
    best_accuracy = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Training
        avg_loss, avg_hierarchical, avg_pair, avg_supcon, avg_ce, train_accuracy = train_joint_epoch(
            model, train_dataloader, hierarchical_criterion, ce_criterion,
            optimizer, device, args.ce_weight
        )

        # Evaluation
        if epoch % args.plot_freq == 0 and not args.no_tsne:
            train_metrics = eval_joint_epoch_with_viz(
                model, train_dataloader, hierarchical_criterion, ce_criterion, device,
                epoch, f"train-{args.train_source}", plot_dir, args.ce_weight
            )
            eval_metrics = eval_joint_epoch_with_viz(
                model, test_dataloader, hierarchical_criterion, ce_criterion, device,
                epoch, f"test-{args.train_source}", plot_dir, args.ce_weight
            )
            eval_metrics2 = eval_joint_epoch_with_viz(
                model, test2_dataloader, hierarchical_criterion, ce_criterion, device,
                epoch, f"test-{external_test_names[0]}", plot_dir, args.ce_weight
            )
            eval_metrics3 = eval_joint_epoch_with_viz(
                model, test3_dataloader, hierarchical_criterion, ce_criterion, device,
                epoch, f"test-{external_test_names[1]}", plot_dir, args.ce_weight
            )
        else:
            train_metrics = eval_joint_epoch(
                model, train_dataloader, hierarchical_criterion, ce_criterion, device, args.ce_weight
            )
            eval_metrics = eval_joint_epoch(
                model, test_dataloader, hierarchical_criterion, ce_criterion, device, args.ce_weight
            )
            eval_metrics2 = eval_joint_epoch(
                model, test2_dataloader, hierarchical_criterion, ce_criterion, device, args.ce_weight
            )
            eval_metrics3 = eval_joint_epoch(
                model, test3_dataloader, hierarchical_criterion, ce_criterion, device, args.ce_weight
            )

        # Logging
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train: Loss={avg_loss:.4f} (Hier={avg_hierarchical:.4f} [Pair={avg_pair:.4f}, SC={avg_supcon:.4f}], CE={avg_ce:.4f})")
            print(f"  Train: Acc={train_accuracy:.2f}%")
            print(f"  Test: Acc={eval_metrics['accuracy']:.2f}%, SilInd={eval_metrics['silhouette_individual']:.4f}, SilPair={eval_metrics['silhouette_pair']:.4f}")
            print(f"  Test: Sep={eval_metrics['separation']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{eval_metrics["accuracy"]:.2f}%'
            })
            epoch_pbar.update(1)

        # Save metrics to CSV
        for metrics, name in [(train_metrics, f"train-{args.train_source}"),
                             (eval_metrics, f"test-{args.train_source}"),
                             (eval_metrics2, f"test-{external_test_names[0]}"),
                             (eval_metrics3, f"test-{external_test_names[1]}")]:
            with open(os.path.join(log_dir, f"{name}_results.csv"), 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_cols)
                writer.writerow(metrics)

        # Save checkpoints
        is_best_silhouette = eval_metrics['silhouette_individual'] > best_silhouette
        if is_best_silhouette:
            best_silhouette = eval_metrics['silhouette_individual']

        is_best_acc = eval_metrics['accuracy'] > best_accuracy
        if is_best_acc:
            best_accuracy = eval_metrics['accuracy']

        save_model_checkpoint(
            model, optimizer, epoch, avg_loss, eval_metrics, args,
            save_dir=os.path.join(log_dir, "checkpoints"),
            is_best=is_best_silhouette,
            is_best_accuracy=is_best_acc
        )

    if args.quiet:
        epoch_pbar.close()

    print("\nTraining complete!")
    print(f"Best silhouette score (individual): {best_silhouette:.4f}")
    print(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
