#!/usr/bin/env python3
"""
Cross-Entropy training with individual lung labels (split CXR model).

This script trains a Siamese network using ONLY cross-entropy classification loss
where each lung gets its true label (normal or nodule).

Key features:
- Uses ImagePairDatasetWithIndividualLabels
- Each lung is labeled individually based on CSV
- Normal pairs: both lungs labeled as normal (0)
- Nodule pairs: one lung labeled nodule (1), other labeled normal (0)
- Classification head on embeddings (not projections)
- No contrastive learning component - pure supervised classification

This provides a baseline for comparing against contrastive learning approaches.
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


def build_transforms(resize_dim: int, single=False):
    """Build transforms for training and testing"""
    if not single:
        base_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])

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

        augment_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    return base_transform, augment_transform


def train_ce_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training loop for pure cross-entropy classification

    Args:
        model: Siamese network model with classification head
        dataloader: DataLoader with individual lung labels
        criterion: CrossEntropyLoss
        optimizer: Optimizer
        device: Device
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        # Unpack batch with individual labels
        img1, img2, label1, label2, pair_class_idx, _path1, _path2 = batch
        img1, img2 = img1.to(device), img2.to(device)
        label1, label2 = label1.to(device), label2.to(device)

        optimizer.zero_grad()

        # Get logits from classification head
        logits1, logits2 = model(img1, img2, return_logits=True)
        all_logits = torch.cat([logits1, logits2], dim=0)
        all_labels = torch.cat([label1, label2], dim=0)

        # Compute CE loss
        loss = criterion(all_logits, all_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(all_logits, 1)
        total += all_labels.size(0)
        correct += (predicted == all_labels).sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def eval_ce_epoch(model, dataloader, criterion, device):
    """
    Evaluation loop with individual lung labels

    Metrics tracked:
    - CE loss
    - Classification accuracy
    - Silhouette score (individual lung clustering quality)
    - Davies-Bouldin score (individual lung clustering quality)
    - Embedding std (collapse detection)
    - Pair distances by pair class
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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

            # Get logits for loss and accuracy
            logits1, logits2 = model(img1, img2, return_logits=True)

            # Get embeddings for metrics (normalized embeddings)
            # For models without projection heads, default forward returns embeddings
            emb1, emb2 = model(img1, img2, return_logits=False)
            all_logits = torch.cat([logits1, logits2], dim=0)
            all_labels = torch.cat([label1, label2], dim=0)

            # Store embeddings with INDIVIDUAL labels for quality metrics
            all_embeddings_list.append(emb1.cpu())
            all_embeddings_list.append(emb2.cpu())
            all_individual_labels_list.append(label1.cpu())
            all_individual_labels_list.append(label2.cpu())

            # Compute loss
            loss = criterion(all_logits, all_labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(all_logits, 1)
            total += all_labels.size(0)
            correct += (predicted == all_labels).sum().item()

            # Calculate distances between pairs (for compatibility with contrastive metrics)
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
    accuracy = 100.0 * correct / total if total > 0 else 0.0

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
        'accuracy': accuracy,
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


def eval_ce_epoch_with_viz(model, dataloader, criterion, device, epoch, testset,
                           plt_path="plots/ce_individual"):
    """Evaluation with visualization"""
    metrics = eval_ce_epoch(model, dataloader, criterion, device)

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

            # Get embeddings for visualization (normalized embeddings)
            # For models without projection heads, default forward returns embeddings
            emb1, emb2 = model(img1, img2, return_logits=False)

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
        save_path=os.path.join(plt_path, f'{testset}_tsne_ce_epoch_{epoch}.png')
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with Cross-Entropy only using individual lung labels (split CXR)"
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
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plot_freq", type=int, default=5)
    parser.add_argument("--plot_dir", type=str, default="plots/ce_individual")
    parser.add_argument("--log_dir", type=str, default="logs/ce_individual")
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
        single=(args.model_name == "single")
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

    # Always use num_classes=2 for binary classification (normal vs nodule)
    num_classes = 2

    # Select model architecture based on spatial flags
    if args.no_single_embedding and args.early_truncation:
        # Use early spatial models (layer3, 14x14 resolution)
        model = helpers.models.SiameseNetworkSpatialEarly(
            backbone,
            embedding_dim=128,
            freeze_backbone=args.freeze_backbone,
            num_classes=num_classes
        ).to(device)
    elif args.no_single_embedding:
        # Use spatial models that process 7x7 feature maps with 1x1 convs
        model = helpers.models.SiameseNetworkSpatial(
            backbone,
            embedding_dim=128,
            freeze_backbone=args.freeze_backbone,
            num_classes=num_classes
        ).to(device)
    else:
        # Use standard models that pool immediately to 2048-dim vectors
        model = helpers.models.SiameseNetwork(
            backbone,
            embedding_dim=128,
            freeze_backbone=args.freeze_backbone,
            num_classes=num_classes
        ).to(device)

    # Loss function (Cross-Entropy only)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup logging directories
    exp_name_parts = [
        args.model_name,
        "ce",  # Add CE identifier
        "frz" if args.freeze_backbone else "unfrz",
        args.process,
        "sym" if args.symmetrical_transforms else "nosym",
        f"bsz{args.bsz}",
        f"lr{args.lr}",
        f"seed{args.seed}",
        str(args.run)
    ]

    # Add "spatial" to name if using spatial (4D) backbone output
    if args.no_single_embedding:
        exp_name_parts.insert(1, "spatial")  # Insert after model_name

    # Add "early" to name if using early truncation (layer3)
    if args.early_truncation:
        exp_name_parts.insert(1, "early")  # Insert after model_name (and spatial if present)

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

    # Log configuration
    config_str = (f"Training with Cross-Entropy only + Individual Lung Labels:\n"
                 f"  model_name={args.model_name}, resize_dim={args.resize_dim}\n"
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
    csv_cols = ['avg_loss', 'accuracy',
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

    # Training loop
    if args.quiet:
        epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress")

    best_silhouette = -float('inf')
    best_accuracy = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Training
        avg_loss, train_accuracy = train_ce_epoch(
            model, train_dataloader, criterion, optimizer, device
        )

        # Evaluation
        if epoch % args.plot_freq == 0 and not args.no_tsne:
            train_metrics = eval_ce_epoch_with_viz(
                model, train_dataloader, criterion, device, epoch,
                f"train-{args.train_source}", plot_dir
            )
            eval_metrics = eval_ce_epoch_with_viz(
                model, test_dataloader, criterion, device, epoch,
                f"test-{args.train_source}", plot_dir
            )
            eval_metrics2 = eval_ce_epoch_with_viz(
                model, test2_dataloader, criterion, device, epoch,
                f"test-{external_test_names[0]}", plot_dir
            )
            eval_metrics3 = eval_ce_epoch_with_viz(
                model, test3_dataloader, criterion, device, epoch,
                f"test-{external_test_names[1]}", plot_dir
            )
        else:
            train_metrics = eval_ce_epoch(
                model, train_dataloader, criterion, device
            )
            eval_metrics = eval_ce_epoch(
                model, test_dataloader, criterion, device
            )
            eval_metrics2 = eval_ce_epoch(
                model, test2_dataloader, criterion, device
            )
            eval_metrics3 = eval_ce_epoch(
                model, test3_dataloader, criterion, device
            )

        # Logging
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train: Loss={avg_loss:.4f}, Accuracy={train_accuracy:.2f}%")
            print(f"  Test: Loss={eval_metrics['avg_loss']:.4f}, Accuracy={eval_metrics['accuracy']:.2f}%")
            print(f"  Test Silhouette: {eval_metrics['silhouette_score']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'Train_Loss': f'{avg_loss:.4f}',
                'Test_Acc': f'{eval_metrics["accuracy"]:.2f}%'
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
        is_best_silhouette = eval_metrics['silhouette_score'] > best_silhouette
        if is_best_silhouette:
            best_silhouette = eval_metrics['silhouette_score']

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
    print(f"Best silhouette score: {best_silhouette:.4f}")
    print(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
