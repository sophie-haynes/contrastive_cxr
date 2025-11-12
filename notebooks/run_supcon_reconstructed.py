#!/usr/bin/env python3
"""
SupCon training script for reconstructed full CXR images from paired lung data.

Reconstructs full CXR images by:
1. Loading lung_l.png and lung_r.png from paired dataset
2. Horizontally flipping lung_r
3. Concatenating lung_l + flipped lung_r horizontally
4. Applying two-view augmentation (standard SupCon)

This ensures the exact same data distribution as paired experiments.
"""
import os
import sys
import csv
import argparse
import math

sys.path.insert(1, '../')
import helpers

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Adding seed stuff
import random
import numpy as np
import torch.nn.functional as F


def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class SupConLossOfficial(torch.nn.Module):
    """
    Official SupCon loss with proper temperature scaling.
    Based on: https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLossOfficial, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: (N, embedding_dim) - L2 normalized embeddings
            labels: (N,) - class labels
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Create mask to exclude self-comparisons
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # Mask out self-comparisons from positive mask
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Official loss formula with temperature scaling
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class ReconstructedCXRWrapper(torch.utils.data.Dataset):
    """
    Wrapper that:
    1. Loads paired lung images (lung_l.png, lung_r.png)
    2. Horizontally flips lung_r
    3. Concatenates them horizontally to reconstruct full CXR
    4. Creates two augmented views for SupCon

    Works directly with PIL images from cache to avoid memory leaks.
    """
    def __init__(self, paired_dataset, transform):
        """
        Args:
            paired_dataset: ImagePairDataset instance (with transform=None to get raw PIL)
            transform: Augmentation transform to apply (should include ToTensor)
        """
        self.paired_dataset = paired_dataset
        self.transform = transform

        # Store pairs directly to avoid repeated dataset access
        self.pairs = paired_dataset.pairs
        self.cache_in_ram = paired_dataset.cache_in_ram
        self._image_cache = paired_dataset._image_cache if paired_dataset.cache_in_ram else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            view1: First augmented view of reconstructed CXR
            view2: Second augmented view of reconstructed CXR
            label: Class label
            path: Pair name (repeated for compatibility)
        """
        pair_info = self.pairs[idx]

        # Load lung images directly (from cache if enabled) - returns PIL RGB
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

        # Horizontally flip lung_r to reconstruct anatomical orientation
        lung_r_flipped = lung_r.transpose(Image.FLIP_LEFT_RIGHT)

        # Concatenate horizontally: lung_l on left, flipped lung_r on right
        width_l, height_l = lung_l.size
        width_r, height_r = lung_r_flipped.size

        # Ensure both have same height (should already be the case)
        if height_l != height_r:
            # Resize to match heights if needed
            target_height = max(height_l, height_r)
            if height_l != target_height:
                lung_l = lung_l.resize((width_l, target_height), Image.BILINEAR)
            if height_r != target_height:
                lung_r_flipped = lung_r_flipped.resize((width_r, target_height), Image.BILINEAR)
            height_l = height_r = target_height

        # Create new image with combined width
        reconstructed_cxr = Image.new('RGB', (width_l + width_r, height_l))
        reconstructed_cxr.paste(lung_l, (0, 0))
        reconstructed_cxr.paste(lung_r_flipped, (width_l, 0))

        # Apply transform twice to create two different views
        view1 = self.transform(reconstructed_cxr)
        view2 = self.transform(reconstructed_cxr)

        pair_name = pair_info['pair_name']
        return view1, view2, pair_info['class_idx'], pair_name, pair_name


def save_model_checkpoint(model, optimizer, epoch, train_loss, eval_metrics, args,
                         save_dir='checkpoints', is_best=False, best_loss=float('inf')):
    """Save model checkpoint with organized structure"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_metrics': eval_metrics,
        'best_loss': best_loss,
        'args': vars(args)
    }

    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✅ New best model saved at epoch {epoch}!")

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and return start epoch and best loss"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
    best_loss = checkpoint.get('best_loss', float('inf'))

    print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    return start_epoch, best_loss


def build_transforms(resize_dim: int, single=False):
    """
    Build transforms for training and testing

    Args:
        resize_dim: Target image size
        single: Whether to use grayscale (for single-channel models)
    """
    if not single:
        # Base transform (no augmentation)
        base_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
        ])

        # Original milder augmentations (matching paired experiments)
        augment_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        # Grayscale transforms
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


def train_supcon_epoch(model, dataloader, criterion, optimizer, device, use_proj=False):
    """Training loop for SupCon loss

    Args:
        use_proj: If True, use projection head outputs for loss
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Reconstructed CXR mode: (view1, view2, labels, path, path)
        view1, view2, labels, _path1, _path2 = batch
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

        optimizer.zero_grad()

        # Get embeddings or projections for both views
        if use_proj:
            # Use projections for loss (discard embeddings during training)
            # forward_one returns projection when model has projection head
            emb1 = model.forward_one(view1, return_embedding=False)
            emb2 = model.forward_one(view2, return_embedding=False)
        else:
            emb1 = model.forward_one(view1)
            emb2 = model.forward_one(view2)

        # Concatenate embeddings and labels
        all_embeddings = torch.cat([emb1, emb2], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)

        # Compute SupCon loss
        loss = criterion(all_embeddings, all_labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def eval_supcon_epoch(model, dataloader, criterion, device, use_proj=False):
    """Evaluation loop with embedding quality metrics

    Args:
        use_proj: If True, use projection head outputs for loss but embeddings for metrics
    """
    model.eval()
    total_loss = 0.0

    # Collect all embeddings for quality metrics
    all_embeddings_list = []
    all_labels_list = []

    # Track augmentation consistency
    view_distances = []

    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    with torch.no_grad():
        for batch in dataloader:
            view1, view2, labels, _path1, _path2 = batch
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

            # Get embeddings (ALWAYS use embeddings for evaluation, not projections)
            if use_proj:
                # For projection models, extract embeddings (not projections) for evaluation
                emb1 = model.forward_one(view1, return_embedding=True)
                emb2 = model.forward_one(view2, return_embedding=True)
                # But compute loss with projections
                proj1 = model.forward_one(view1, return_embedding=False)
                proj2 = model.forward_one(view2, return_embedding=False)
            else:
                emb1 = model.forward_one(view1)
                emb2 = model.forward_one(view2)
                proj1, proj2 = emb1, emb2

            # Store for quality metrics (only one embedding per image to avoid duplication)
            all_embeddings_list.append(emb1.cpu())
            all_labels_list.append(labels.cpu())

            # Compute loss (using projections if use_proj=True)
            all_proj = torch.cat([proj1, proj2], dim=0)
            all_labels_batch = torch.cat([labels, labels], dim=0)

            # Calculate consistency between augmented views (lower = better)
            distances = F.pairwise_distance(emb1, emb2, p=2)
            view_distances.extend(distances.cpu().numpy())

            loss = criterion(all_proj, all_labels_batch)
            total_loss += loss.item()

    # Concatenate all embeddings
    all_embeddings_np = torch.cat(all_embeddings_list, dim=0).numpy()
    all_labels_np = torch.cat(all_labels_list, dim=0).numpy()

    # Calculate basic metrics
    avg_loss = total_loss / max(len(dataloader), 1)

    # Calculate embedding quality metrics
    silhouette = silhouette_score(all_embeddings_np, all_labels_np) if len(np.unique(all_labels_np)) > 1 else 0
    davies_bouldin = davies_bouldin_score(all_embeddings_np, all_labels_np) if len(np.unique(all_labels_np)) > 1 else 0

    # Embedding std (collapse detection)
    embedding_std = np.std(all_embeddings_np)

    # Augmentation consistency
    aug_consistency = np.mean(view_distances) if view_distances else 0

    metrics = {
        'avg_loss': avg_loss,
        'aug_consistency': aug_consistency,  # Lower = better invariance
        'num_samples': len(view_distances),
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'embedding_std': embedding_std
    }

    return metrics


def eval_supcon_epoch_with_viz(model, dataloader, criterion, device, epoch, testset,
                               plt_path="plots/subsets", use_proj=False):
    """Evaluation with visualization"""
    metrics = eval_supcon_epoch(model, dataloader, criterion, device, use_proj)

    # t-SNE plot
    helpers.viz.generate_tsne_plot_fullimage(
        model, dataloader, device,
        epoch=epoch,
        save_path=os.path.join(plt_path, f'{testset}_tsne_epoch_{epoch}.png')
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SupCon with reconstructed full CXR from paired lungs"
    )
    parser.add_argument("--model_name", type=str,
                       choices=["rgb", "grey", "single", "rad", "randinit"],
                       required=True)
    parser.add_argument("--resize_dim", type=int, default=224)

    parser.add_argument("--paired_dataset_path", type=str, default="../split_node21_sets",
                       help="Path to paired dataset")
    parser.add_argument("--process", type=str,
                       choices=["lung_seg", "crop", "arch_seg"],
                       required=True)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--cache_in_ram", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")

    # Loss configuration
    parser.add_argument("--temperature", type=float, default=0.07,
                       help="Temperature for SupCon loss")
    parser.add_argument("--base_temperature", type=float, default=0.07,
                       help="Base temperature for loss scaling")

    # Projection head option
    parser.add_argument("--use_proj", action="store_true",
                       help="Use SiameseNetworkWithProjection (SimCLR-style projection head)")
    parser.add_argument("--projection_dim", type=int, default=128,
                       help="Projection head output dimension (default: 128)")

    # Optimizer configuration
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)

    # Logging
    parser.add_argument("--plot_freq", type=int, default=5)
    parser.add_argument("--plot_dir", type=str, default="plots/subsets")
    parser.add_argument("--log_dir", type=str, default="logs/subsets")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument('-q', '--quiet', action="store_true")
    parser.add_argument("--train_source", type=str,
                       choices=["chestxray14", "jsrt", "padchest"],
                       required=True)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)

    # Resume training
    parser.add_argument("--resume", type=str, default="",
                       help="Path to checkpoint to resume from")

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
    train_path = os.path.join(args.paired_dataset_path, args.process, args.train_source, "train")
    test_path = os.path.join(args.paired_dataset_path, args.process, args.train_source, "test")

    external_test_names.remove(args.train_source)
    test2_path = os.path.join(args.paired_dataset_path, args.process, external_test_names[0], "test")
    test3_path = os.path.join(args.paired_dataset_path, args.process, external_test_names[1], "test")

    # Load paired datasets WITHOUT transforms (we'll apply them in ReconstructedCXRWrapper)
    train_base = helpers.dataloading.ImagePairDataset(
        root=train_path,
        transform=None,  # No transform - ReconstructedCXRWrapper will apply augment_transform
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    train_dataset = ReconstructedCXRWrapper(train_base, augment_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bsz,
                                 shuffle=True, num_workers=args.workers)

    test_base = helpers.dataloading.ImagePairDataset(
        root=test_path,
        transform=None,  # No transform - ReconstructedCXRWrapper will apply base_transform
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    test_dataset = ReconstructedCXRWrapper(test_base, base_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bsz,
                                 shuffle=False, num_workers=args.workers)

    test2_base = helpers.dataloading.ImagePairDataset(
        root=test2_path,
        transform=None,
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    test2_dataset = ReconstructedCXRWrapper(test2_base, base_transform)
    test2_dataloader = DataLoader(test2_dataset, batch_size=args.bsz,
                                  shuffle=False, num_workers=args.workers)

    test3_base = helpers.dataloading.ImagePairDataset(
        root=test3_path,
        transform=None,
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    test3_dataset = ReconstructedCXRWrapper(test3_base, base_transform)
    test3_dataloader = DataLoader(test3_dataset, batch_size=args.bsz,
                                  shuffle=False, num_workers=args.workers)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    backbone = helpers.models.load_truncated_model(args.model_name)

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

    # Loss function
    criterion = SupConLossOfficial(
        temperature=args.temperature,
        base_temperature=args.base_temperature
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint if resuming
    start_epoch = 0
    best_silhouette = -float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer, device)
        # Note: best_loss from checkpoint is for compatibility, but we track silhouette now

    # Setup logging directories
    exp_name_parts = [
        args.model_name,
        "frz" if args.freeze_backbone else "unfrz",
        "reconstructed",
        args.process,
        f"bsz{args.bsz}",
        f"lr{args.lr}",
        f"temp{args.temperature}",
        f"seed{args.seed}",
        str(args.run)
    ]

    # Add "proj" to name if using projection head
    if args.use_proj:
        exp_name_parts.insert(1, "proj")  # Insert after model_name

    exp_name = "_".join(exp_name_parts)

    if args.comment != "":
        exp_name = "_".join([exp_name, args.comment])

    plot_dir = os.path.join(args.plot_dir, args.train_source, "supcon", "reconstructed", exp_name)
    log_dir = os.path.join(args.log_dir, args.train_source, "supcon", "reconstructed", exp_name)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Log configuration
    config_str = (f"Training SupCon (Reconstructed CXR from paired lungs):\n"
                 f"  model_name={args.model_name}, resize_dim={args.resize_dim}\n"
                 f"  process={args.process}, train_source={args.train_source}\n"
                 f"  temperature={args.temperature}, base_temperature={args.base_temperature}\n"
                 f"  use_proj={args.use_proj}, projection_dim={args.projection_dim}\n"
                 f"  bsz={args.bsz}, lr={args.lr}, epochs={args.epochs}\n"
                 f"  seed={args.seed}, comment={args.comment}\n"
                 f"  device={device}")

    print(config_str)

    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(config_str)

    # Setup CSV logging
    csv_cols = ['epoch', 'lr', 'avg_loss', 'aug_consistency', 'num_samples',
                'silhouette_score', 'davies_bouldin_score', 'embedding_std']

    # Only write headers if not resuming
    if not args.resume:
        for dataset_name in [f"train-{args.train_source}",
                            f"test-{args.train_source}",
                            f"test-{external_test_names[0]}",
                            f"test-{external_test_names[1]}"]:
            with open(os.path.join(log_dir, f"{dataset_name}_results.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_cols)
                writer.writeheader()

    # Training loop
    if args.quiet:
        epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress", initial=start_epoch, total=args.epochs)

    for epoch in range(start_epoch, args.epochs):
        # Training (constant LR - no warmup/decay for fair comparison)
        avg_loss = train_supcon_epoch(
            model, train_dataloader, criterion, optimizer, device, args.use_proj
        )

        # Evaluation
        if epoch % args.plot_freq == 0:
            train_metrics = eval_supcon_epoch_with_viz(
                model, train_dataloader, criterion, device, epoch,
                f"train-{args.train_source}", plot_dir, args.use_proj
            )
            eval_metrics = eval_supcon_epoch_with_viz(
                model, test_dataloader, criterion, device, epoch,
                f"test-{args.train_source}", plot_dir, args.use_proj
            )
            eval_metrics2 = eval_supcon_epoch_with_viz(
                model, test2_dataloader, criterion, device, epoch,
                f"test-{external_test_names[0]}", plot_dir, args.use_proj
            )
            eval_metrics3 = eval_supcon_epoch_with_viz(
                model, test3_dataloader, criterion, device, epoch,
                f"test-{external_test_names[1]}", plot_dir, args.use_proj
            )
        else:
            train_metrics = eval_supcon_epoch(
                model, train_dataloader, criterion, device, args.use_proj
            )
            eval_metrics = eval_supcon_epoch(
                model, test_dataloader, criterion, device, args.use_proj
            )
            eval_metrics2 = eval_supcon_epoch(
                model, test2_dataloader, criterion, device, args.use_proj
            )
            eval_metrics3 = eval_supcon_epoch(
                model, test3_dataloader, criterion, device, args.use_proj
            )

        # Add epoch and lr to metrics
        for metrics in [train_metrics, eval_metrics, eval_metrics2, eval_metrics3]:
            metrics['epoch'] = epoch
            metrics['lr'] = args.lr  # Constant LR

        # Logging
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {args.lr:.6f}")
            print(f"  Train: Loss={avg_loss:.4f}")
            print(f"  Train Silhouette: {train_metrics['silhouette_score']:.4f}")
            print(f"  Test Silhouette: {eval_metrics['silhouette_score']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'LR': f'{args.lr:.6f}',
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

        # Save checkpoint (track best based on silhouette score)
        is_best = eval_metrics['silhouette_score'] > best_silhouette
        if is_best:
            best_silhouette = eval_metrics['silhouette_score']

        save_model_checkpoint(
            model, optimizer, epoch, avg_loss, eval_metrics, args,
            save_dir=os.path.join(log_dir, "checkpoints"),
            is_best=is_best,
            best_loss=eval_metrics['avg_loss']  # Still save loss for compatibility
        )

    if args.quiet:
        epoch_pbar.close()

    print("\nTraining complete!")
    print(f"Best silhouette score: {best_silhouette:.4f}")


if __name__ == "__main__":
    main()
