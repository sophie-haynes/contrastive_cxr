#!/usr/bin/env python3
"""
Hierarchical loss training with Prototypes and individual lung labels.

This script extends run_hierarchical_individual_labels.py by incorporating supervised
prototypes from "A Tale of Two Classes" (Mildenberger et al., CVPR 2025).

Key features:
- Uses ImagePairDatasetWithIndividualLabels
- Each lung labeled individually (normal=0, nodule=1)
- HierarchicalLossWithPrototypes: pair loss + SupCon with prototypes
- Asymmetric margins (eps_0, eps_1) to help minority class form tighter clusters

Prototypes are computed once at the start of training from the initial embeddings,
providing stable reference points that help the minority class (nodules) form more
compact clusters and generalize better to unseen distributions.
"""
import os
import sys
import csv
import argparse

sys.path.insert(1, '../')
import helpers

import torch
from torchvision import transforms
from tqdm import tqdm

# Import the Hierarchical loss with prototypes
from helpers.supcon import HierarchicalLossWithPrototypes, compute_prototypes_from_embeddings

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


def save_model_checkpoint(model, optimizer, epoch, train_loss, eval_metrics, args,
                         save_dir='checkpoints', is_best=False):
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
        print(f"New best model saved at epoch {epoch}!")

    return checkpoint_path


def build_transforms(resize_dim: int, single=False, aggressive_aug=False):
    """
    Build transforms for training and testing

    Args:
        resize_dim: Target image size
        single: Whether to use grayscale (for single-channel models)
        aggressive_aug: Use official SupContrast augmentations (stronger)
    """
    if not single:
        base_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])

        if aggressive_aug:
            # Official SupContrast augmentations
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
            # Original milder augmentations
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


def compute_prototypes(model, dataloader, device, use_proj=False):
    """
    Compute prototypes from training data embeddings.

    Prototypes are found by optimizing for the direction that minimizes
    cosine similarity to all training embeddings, providing an ideal
    separation axis for the two classes.

    Args:
        model: Siamese network model
        dataloader: Training dataloader
        device: Device
        use_proj: If True, extract embeddings (not projections)

    Returns:
        prototypes: Tensor of shape [2, embedding_dim]
    """
    print("\n" + "="*60)
    print("COMPUTING PROTOTYPES FROM TRAINING DATA")
    print("="*60)

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
            img1, img2 = img1.to(device), img2.to(device)

            # Always extract embeddings (not projections) for prototype computation
            if use_proj:
                emb1, emb2 = model(img1, img2, return_embedding=True)
            else:
                emb1, emb2 = model(img1, img2)

            all_embeddings.append(emb1.cpu())
            all_embeddings.append(emb2.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Collected {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")

    # Compute prototypes via optimization
    prototypes = compute_prototypes_from_embeddings(
        all_embeddings,
        n_iterations=10000,
        lr=0.0001,
        verbose=True
    )

    print("="*60 + "\n")
    model.train()
    return prototypes


def train_hierarchical_proto_epoch(model, dataloader, criterion, optimizer, device, use_proj=False):
    """
    Training loop for Hierarchical loss with prototypes and individual lung labels

    Args:
        model: Siamese network model
        dataloader: DataLoader with individual lung labels
        criterion: HierarchicalLossWithPrototypes
        optimizer: Optimizer
        device: Device
        use_proj: If True, use projection head outputs for loss
    """
    model.train()
    total_loss = 0.0
    total_pair_loss = 0.0
    total_supcon_loss = 0.0

    for batch in dataloader:
        # Unpack batch with individual labels
        img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
        img1, img2 = img1.to(device), img2.to(device)
        label1, label2 = label1.to(device), label2.to(device)
        pair_class_idx = pair_class_idx.to(device)

        optimizer.zero_grad()

        # Get embeddings or projections
        if use_proj:
            # Use projections for loss (discard embeddings during training)
            emb1, emb2 = model(img1, img2, return_embedding=False)
        else:
            emb1, emb2 = model(img1, img2)

        # Hierarchical loss with prototypes
        # pair_class_idx: pair-level labels for contrastive pair loss
        # label1, label2: individual lung labels for SupCon loss with prototypes
        loss, pair_loss, supcon_loss = criterion(emb1, emb2, pair_class_idx, label1, label2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pair_loss += pair_loss.item()
        total_supcon_loss += supcon_loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_pair_loss = total_pair_loss / max(len(dataloader), 1)
    avg_supcon_loss = total_supcon_loss / max(len(dataloader), 1)

    return avg_loss, avg_pair_loss, avg_supcon_loss


def eval_hierarchical_proto_epoch(model, dataloader, criterion, device, use_proj=False):
    """
    Evaluation loop with individual lung labels and prototypes

    Metrics tracked:
    - Hierarchical loss (pair + supcon with prototypes and individual labels)
    - Pair loss component
    - SupCon loss component
    - Silhouette score (individual lung clustering quality)
    - Davies-Bouldin score (individual lung clustering quality)
    - Embedding std (collapse detection)
    - Pair distances by pair class
    """
    model.eval()
    total_loss = 0.0
    total_pair_loss = 0.0
    total_supcon_loss = 0.0

    # Track pair distances by pair class
    normal_pair_distances = []
    nodule_pair_distances = []

    # Collect all individual lung embeddings with their individual labels
    all_embeddings_list = []
    all_individual_labels_list = []

    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch with individual labels
            img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
            img1, img2 = img1.to(device), img2.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            pair_class_idx = pair_class_idx.to(device)

            # Get embeddings (ALWAYS use embeddings for evaluation, not projections)
            if use_proj:
                # For projection models, extract embeddings (not projections) for evaluation
                emb1, emb2 = model(img1, img2, return_embedding=True)
                # But compute loss with projections
                proj1, proj2 = model(img1, img2, return_embedding=False)
            else:
                emb1, emb2 = model(img1, img2)
                proj1, proj2 = emb1, emb2

            # Store embeddings with INDIVIDUAL labels for quality metrics
            all_embeddings_list.append(emb1.cpu())
            all_embeddings_list.append(emb2.cpu())
            all_individual_labels_list.append(label1.cpu())
            all_individual_labels_list.append(label2.cpu())

            # Compute loss with individual labels (using projections if use_proj=True)
            loss, pair_loss, supcon_loss = criterion(proj1, proj2, pair_class_idx, label1, label2)
            total_loss += loss.item()
            total_pair_loss += pair_loss.item()
            total_supcon_loss += supcon_loss.item()

            # Calculate distances between pairs (for pair distance metrics)
            distances = F.pairwise_distance(emb1, emb2, p=2)

            # Separate by PAIR class
            normal_pair_mask = (pair_class_idx == 0)
            nodule_pair_mask = (pair_class_idx == 1)

            if normal_pair_mask.sum() > 0:
                normal_pair_distances.extend(distances[normal_pair_mask].cpu().numpy())
            if nodule_pair_mask.sum() > 0:
                nodule_pair_distances.extend(distances[nodule_pair_mask].cpu().numpy())

    # Concatenate all embeddings with individual labels
    all_embeddings_np = torch.cat(all_embeddings_list, dim=0).numpy()
    all_individual_labels_np = torch.cat(all_individual_labels_list, dim=0).numpy()

    # Calculate basic metrics
    avg_loss = total_loss / max(len(dataloader), 1)
    avg_pair_loss = total_pair_loss / max(len(dataloader), 1)
    avg_supcon_loss = total_supcon_loss / max(len(dataloader), 1)

    normal_pair_mean = np.mean(normal_pair_distances) if normal_pair_distances else 0
    nodule_pair_mean = np.mean(nodule_pair_distances) if nodule_pair_distances else 0
    separation = nodule_pair_mean - normal_pair_mean

    # Calculate embedding quality metrics using INDIVIDUAL labels
    silhouette = silhouette_score(all_embeddings_np, all_individual_labels_np) if len(np.unique(all_individual_labels_np)) > 1 else 0
    davies_bouldin = davies_bouldin_score(all_embeddings_np, all_individual_labels_np) if len(np.unique(all_individual_labels_np)) > 1 else 0

    # Embedding std (collapse detection)
    embedding_std = np.std(all_embeddings_np)

    # Count individual lung labels
    num_normal_lungs = np.sum(all_individual_labels_np == 0)
    num_nodule_lungs = np.sum(all_individual_labels_np == 1)

    metrics = {
        'avg_loss': avg_loss,
        'pair_loss': avg_pair_loss,
        'supcon_loss': avg_supcon_loss,
        'normal_pair_dist_mean': normal_pair_mean,
        'nodule_pair_dist_mean': nodule_pair_mean,
        'separation': separation,
        'num_normal_pairs': len(normal_pair_distances),
        'num_nodule_pairs': len(nodule_pair_distances),
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'embedding_std': embedding_std,
        'num_normal_lungs': num_normal_lungs,
        'num_nodule_lungs': num_nodule_lungs
    }

    return metrics


def eval_hierarchical_proto_epoch_with_viz(model, dataloader, criterion, device, epoch, testset,
                                           plt_path="plots/hierarchical_proto_individual", use_proj=False):
    """Evaluation with visualization"""
    metrics = eval_hierarchical_proto_epoch(model, dataloader, criterion, device, use_proj)

    import numpy as np

    # Collect embeddings and labels for visualization
    all_embeddings_list = []
    all_individual_labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
            img1, img2 = img1.to(device), img2.to(device)
            label1, label2 = label1.to(device), label2.to(device)

            # Always use embeddings for visualization (not projections)
            if use_proj:
                emb1, emb2 = model(img1, img2, return_embedding=True)
            else:
                emb1, emb2 = model(img1, img2)

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
        save_path=os.path.join(plt_path, f'{testset}_tsne_proto_epoch_{epoch}.png')
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with Hierarchical loss + Prototypes using individual lung labels"
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
                       help="Weight for pair contrastive loss")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="Weight for SupCon loss")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin for pair contrastive loss")
    parser.add_argument("--distance", type=str, default="euclidean",
                       choices=["euclidean", "cosine"],
                       help="Distance metric for pair loss")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for SupCon loss")

    # Prototype configuration
    parser.add_argument("--eps_0", type=float, default=0.3,
                       help="Margin for class 0 (majority/normal). Lower = less pulling to prototype")
    parser.add_argument("--eps_1", type=float, default=0.5,
                       help="Margin for class 1 (minority/nodule). Higher = more pulling to prototype")
    parser.add_argument("--recompute_prototypes_every", type=int, default=0,
                       help="Recompute prototypes every N epochs (0 = compute once at start)")

    # Projection head option
    parser.add_argument("--use_proj", action="store_true",
                       help="Use SiameseNetworkWithProjection (SimCLR-style projection head)")
    parser.add_argument("--projection_dim", type=int, default=128,
                       help="Projection head output dimension (default: 128)")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plot_freq", type=int, default=5)
    parser.add_argument("--plot_dir", type=str, default="plots/hierarchical_proto_individual")
    parser.add_argument("--log_dir", type=str, default="logs/hierarchical_proto_individual")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument('-q', '--quiet', action="store_true")
    parser.add_argument("--train_source", type=str,
                       choices=["chestxray14", "jsrt", "padchest"],
                       required=True)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--no_tsne', action="store_true")
    parser.add_argument('--no_single_embedding', action="store_true",
                       help="Use 4D backbone output (B, 2048, 7, 7) instead of 2D (B, 2048)")
    parser.add_argument('--early_truncation', action="store_true",
                       help="Use layer3 truncation for higher spatial resolution (B, 1024, 14, 14). Requires --no_single_embedding")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None,
                       help="Specific epoch to resume from (will look for checkpoint_epoch_X.pth)")

    return parser.parse_args()


def main():
    args = parse_args()

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

    print("\nLoading external test data 1 with individual lung labels...")
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

    print("\nLoading external test data 2 with individual lung labels...")
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
            model = helpers.models.SiameseNetworkWithProjectionSpatialEarly(
                backbone,
                embedding_dim=128,
                projection_dim=args.projection_dim,
                freeze_backbone=args.freeze_backbone
            ).to(device)
        else:
            model = helpers.models.SiameseNetworkSpatialEarly(
                backbone,
                embedding_dim=128,
                freeze_backbone=args.freeze_backbone
            ).to(device)
    elif args.no_single_embedding:
        # Use spatial models that process 7x7 feature maps with 1x1 convs
        if args.use_proj:
            model = helpers.models.SiameseNetworkWithProjectionSpatial(
                backbone,
                embedding_dim=128,
                projection_dim=args.projection_dim,
                freeze_backbone=args.freeze_backbone
            ).to(device)
        else:
            model = helpers.models.SiameseNetworkSpatial(
                backbone,
                embedding_dim=128,
                freeze_backbone=args.freeze_backbone
            ).to(device)
    else:
        # Use standard models that pool immediately to 2048-dim vectors
        if args.use_proj:
            model = helpers.models.SiameseNetworkWithProjection(
                backbone,
                embedding_dim=128,
                projection_dim=args.projection_dim,
                freeze_backbone=args.freeze_backbone
            ).to(device)
        else:
            model = helpers.models.SiameseNetwork(
                backbone,
                embedding_dim=128,
                freeze_backbone=args.freeze_backbone
            ).to(device)

    # Loss function (Hierarchical with Prototypes)
    criterion = HierarchicalLossWithPrototypes(
        alpha=args.alpha,
        beta=args.beta,
        margin=args.margin,
        distance=args.distance,
        temperature=args.temperature,
        eps_0=args.eps_0,
        eps_1=args.eps_1,
        minority_class=1  # Nodule is minority class
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup logging directories
    exp_name_parts = [
        args.model_name,
        "frz" if args.freeze_backbone else "unfrz",
        args.process,
        "sym" if args.symmetrical_transforms else "nosym",
        f"bsz{args.bsz}",
        f"lr{args.lr}",
        f"alpha{args.alpha}_beta{args.beta}",
        f"temp{args.temperature}",
        f"eps0-{args.eps_0}_eps1-{args.eps_1}",  # Include prototype margins
        f"seed{args.seed}",
        str(args.run)
    ]

    # Add "proto" to name to distinguish from non-prototype runs
    exp_name_parts.insert(1, "proto")

    # Add "proj" to name if using projection head
    if args.use_proj:
        exp_name_parts.insert(1, "proj")  # Insert after model_name

    # Add "spatial" to name if using spatial (4D) backbone output
    if args.no_single_embedding:
        exp_name_parts.insert(1, "spatial")  # Insert after model_name (and proj if present)

    # Add "early" to name if using early truncation (layer3)
    if args.early_truncation:
        exp_name_parts.insert(1, "early")  # Insert after model_name (and proj/spatial if present)

    # Add "aggaug" if using aggressive augmentations
    if args.aggressive_aug:
        exp_name_parts.insert(1, "aggaug")

    exp_name = "_".join(exp_name_parts)

    # Handle checkpoint resumption
    start_epoch = 0
    if args.resume or args.resume_epoch is not None:
        if args.resume:
            checkpoint_path = args.resume
        else:
            # Build checkpoint path from resume_epoch
            log_dir_temp = os.path.join(args.log_dir, args.train_source, exp_name)
            checkpoint_path = os.path.join(log_dir_temp, "checkpoints", f"checkpoint_epoch_{args.resume_epoch}.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Start from next epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Previous train loss: {checkpoint['train_loss']:.4f}")
        if 'eval_metrics' in checkpoint:
            print(f"Previous eval metrics: {checkpoint['eval_metrics']}")
        print()

    if args.comment != "":
        exp_name = "_".join([exp_name, args.comment])

    plot_dir = os.path.join(args.plot_dir, args.train_source, exp_name)
    log_dir = os.path.join(args.log_dir, args.train_source, exp_name)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # COMPUTE PROTOTYPES BEFORE TRAINING
    # Only compute if not resuming OR if it's time to recompute
    if start_epoch == 0:
        prototypes = compute_prototypes(model, train_dataloader, device, args.use_proj)
        criterion.set_prototypes(prototypes)

        # Save prototypes for future reference
        torch.save(prototypes, os.path.join(log_dir, "prototypes.pth"))

    # Log configuration
    config_str = (f"Training with Hierarchical loss + Prototypes + Individual Lung Labels:\n"
                 f"  model_name={args.model_name}, resize_dim={args.resize_dim}\n"
                 f"  alpha={args.alpha}, beta={args.beta}\n"
                 f"  margin={args.margin}, distance={args.distance}\n"
                 f"  temperature={args.temperature}\n"
                 f"  eps_0={args.eps_0}, eps_1={args.eps_1}\n"
                 f"  recompute_prototypes_every={args.recompute_prototypes_every}\n"
                 f"  use_proj={args.use_proj}, projection_dim={args.projection_dim}\n"
                 f"  bsz={args.bsz}, lr={args.lr}, epochs={args.epochs}\n"
                 f"  label_csv={args.label_csv}\n"
                 f"  seed={args.seed}, comment={args.comment}\n"
                 f"  device={device}")

    print("\n" + "="*60)
    print(config_str)
    print("="*60 + "\n")

    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(config_str)

    # Setup CSV logging
    csv_cols = ['avg_loss', 'pair_loss', 'supcon_loss',
                'normal_pair_dist_mean', 'nodule_pair_dist_mean',
                'separation', 'num_normal_pairs', 'num_nodule_pairs',
                'silhouette_score', 'davies_bouldin_score', 'embedding_std',
                'num_normal_lungs', 'num_nodule_lungs']

    # Only create new CSV files if not resuming
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
        # Load prototypes if resuming
        prototypes_path = os.path.join(log_dir, "prototypes.pth")
        if os.path.exists(prototypes_path):
            prototypes = torch.load(prototypes_path)
            criterion.set_prototypes(prototypes)
            print(f"Loaded prototypes from {prototypes_path}")

    # Training loop
    if args.quiet:
        epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress")

    best_silhouette = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Recompute prototypes if requested
        if args.recompute_prototypes_every > 0 and epoch > 0 and epoch % args.recompute_prototypes_every == 0:
            print(f"\nRecomputing prototypes at epoch {epoch}...")
            prototypes = compute_prototypes(model, train_dataloader, device, args.use_proj)
            criterion.set_prototypes(prototypes)
            torch.save(prototypes, os.path.join(log_dir, f"prototypes_epoch_{epoch}.pth"))

        # Training
        avg_loss, avg_pair_loss, avg_supcon_loss = train_hierarchical_proto_epoch(
            model, train_dataloader, criterion, optimizer, device, args.use_proj
        )

        # Evaluation
        if epoch % args.plot_freq == 0 and not args.no_tsne:
            train_metrics = eval_hierarchical_proto_epoch_with_viz(
                model, train_dataloader, criterion, device, epoch,
                f"train-{args.train_source}", plot_dir, args.use_proj
            )
            eval_metrics = eval_hierarchical_proto_epoch_with_viz(
                model, test_dataloader, criterion, device, epoch,
                f"test-{args.train_source}", plot_dir, args.use_proj
            )
            eval_metrics2 = eval_hierarchical_proto_epoch_with_viz(
                model, test2_dataloader, criterion, device, epoch,
                f"test-{external_test_names[0]}", plot_dir, args.use_proj
            )
            eval_metrics3 = eval_hierarchical_proto_epoch_with_viz(
                model, test3_dataloader, criterion, device, epoch,
                f"test-{external_test_names[1]}", plot_dir, args.use_proj
            )
        else:
            train_metrics = eval_hierarchical_proto_epoch(
                model, train_dataloader, criterion, device, args.use_proj
            )
            eval_metrics = eval_hierarchical_proto_epoch(
                model, test_dataloader, criterion, device, args.use_proj
            )
            eval_metrics2 = eval_hierarchical_proto_epoch(
                model, test2_dataloader, criterion, device, args.use_proj
            )
            eval_metrics3 = eval_hierarchical_proto_epoch(
                model, test3_dataloader, criterion, device, args.use_proj
            )

        # Logging
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train: Loss={avg_loss:.4f} (Pair={avg_pair_loss:.4f}, SupCon={avg_supcon_loss:.4f})")
            print(f"  Train Silhouette: {train_metrics['silhouette_score']:.4f}")
            print(f"  Test Silhouette: {eval_metrics['silhouette_score']:.4f}")
            print(f"  Train Sep: {train_metrics['separation']:.4f}")
            print(f"  Test Sep: {eval_metrics['separation']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'Train_Loss': f'{avg_loss:.4f}',
                'Test_Sil': f'{eval_metrics["silhouette_score"]:.4f}'
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

        # Save checkpoint (using silhouette score as best metric)
        is_best = eval_metrics['silhouette_score'] > best_silhouette
        if is_best:
            best_silhouette = eval_metrics['silhouette_score']

        save_model_checkpoint(
            model, optimizer, epoch, avg_loss, eval_metrics, args,
            save_dir=os.path.join(log_dir, "checkpoints"),
            is_best=is_best
        )

    if args.quiet:
        epoch_pbar.close()

    print("\nTraining complete!")
    print(f"Best silhouette score: {best_silhouette:.4f}")


if __name__ == "__main__":
    main()
