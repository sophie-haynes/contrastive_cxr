#!/usr/bin/env python3
"""
Cross-entropy classification baseline for reconstructed full CXR images.

This script trains a standard supervised classifier (no contrastive learning)
on reconstructed CXR images (lung_l + flipped lung_r concatenated).
Serves as a baseline to compare against contrastive learning approaches.
"""
import os
import sys
import csv
import argparse
import random
import numpy as np

sys.path.insert(1, '../')
import helpers

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class ReconstructedCXRDataset(torch.utils.data.Dataset):
    """
    Dataset that reconstructs full CXR images from paired lung data.
    1. Loads lung_l.png and lung_r.png
    2. Horizontally flips lung_r
    3. Concatenates them horizontally
    4. Returns single image with label
    """
    def __init__(self, paired_dataset, transform):
        """
        Args:
            paired_dataset: ImagePairDataset instance (with transform=None)
            transform: Transform to apply to reconstructed image
        """
        self.paired_dataset = paired_dataset
        self.transform = transform
        self.pairs = paired_dataset.pairs
        self.cache_in_ram = paired_dataset.cache_in_ram
        self._image_cache = paired_dataset._image_cache if paired_dataset.cache_in_ram else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            image: Reconstructed CXR image tensor
            label: Class label (0=normal, 1=nodule)
        """
        pair_info = self.pairs[idx]

        # Load lung images (from cache if enabled)
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

        # Concatenate horizontally
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

        # Apply transform
        image = self.transform(reconstructed_cxr)
        label = pair_info['class_idx']

        return image, label


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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    preds_array = np.array(all_preds)
    labels_array = np.array(all_labels)
    probs_array = np.array(all_probs)

    metrics = {
        'avg_loss': avg_loss,
        'accuracy': accuracy_score(labels_array, preds_array),
        'precision': precision_score(labels_array, preds_array, zero_division=0),
        'recall': recall_score(labels_array, preds_array, zero_division=0),
        'f1': f1_score(labels_array, preds_array, zero_division=0),
        'auc': roc_auc_score(labels_array, probs_array) if len(np.unique(labels_array)) > 1 else 0.0
    }

    return metrics


def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluation loop"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    preds_array = np.array(all_preds)
    labels_array = np.array(all_labels)
    probs_array = np.array(all_probs)

    metrics = {
        'avg_loss': avg_loss,
        'accuracy': accuracy_score(labels_array, preds_array),
        'precision': precision_score(labels_array, preds_array, zero_division=0),
        'recall': recall_score(labels_array, preds_array, zero_division=0),
        'f1': f1_score(labels_array, preds_array, zero_division=0),
        'auc': roc_auc_score(labels_array, probs_array) if len(np.unique(labels_array)) > 1 else 0.0
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, args, save_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }

    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved!")

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and return start epoch"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-entropy classification baseline for reconstructed CXR"
    )
    parser.add_argument("--model_name", type=str,
                       choices=["rgb", "grey", "single", "rad", "randinit"],
                       required=True)
    parser.add_argument("--resize_dim", type=int, default=224)

    parser.add_argument("--paired_dataset_path", type=str, default="../split_node21_sets")
    parser.add_argument("--process", type=str,
                       choices=["lung_seg", "crop", "arch_seg"],
                       required=True)
    parser.add_argument("--train_source", type=str,
                       choices=["chestxray14", "jsrt", "padchest"],
                       required=True)

    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--cache_in_ram", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--log_dir", type=str, default="logs/subsets")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument('-q', '--quiet', action="store_true")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--resume", type=str, default="",
                       help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()
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

    # Load paired datasets
    train_base = helpers.dataloading.ImagePairDataset(
        root=train_path,
        transform=None,
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    train_dataset = ReconstructedCXRDataset(train_base, augment_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bsz,
                                 shuffle=True, num_workers=args.workers)

    test_base = helpers.dataloading.ImagePairDataset(
        root=test_path,
        transform=None,
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    test_dataset = ReconstructedCXRDataset(test_base, base_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bsz,
                                 shuffle=False, num_workers=args.workers)

    test2_base = helpers.dataloading.ImagePairDataset(
        root=test2_path,
        transform=None,
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    test2_dataset = ReconstructedCXRDataset(test2_base, base_transform)
    test2_dataloader = DataLoader(test2_dataset, batch_size=args.bsz,
                                  shuffle=False, num_workers=args.workers)

    test3_base = helpers.dataloading.ImagePairDataset(
        root=test3_path,
        transform=None,
        class_to_idx={'nodule': 1, 'normal': 0},
        cache_in_ram=args.cache_in_ram
    )
    test3_dataset = ReconstructedCXRDataset(test3_base, base_transform)
    test3_dataloader = DataLoader(test3_dataset, batch_size=args.bsz,
                                  shuffle=False, num_workers=args.workers)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model - use load_full_model which handles everything
    model = helpers.models.load_full_model(
        model_name=args.model_name,
        num_classes=2,
        freeze_backbone=args.freeze_backbone,
        device=device
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)

    # Setup logging
    exp_name = "_".join([
        args.model_name,
        "frz" if args.freeze_backbone else "unfrz",
        "ce_reconstructed",
        args.process,
        f"bsz{args.bsz}",
        f"lr{args.lr}",
        f"seed{args.seed}",
        str(args.run)
    ])

    if args.comment != "":
        exp_name = "_".join([exp_name, args.comment])

    log_dir = os.path.join(args.log_dir, args.train_source, "ce", "reconstructed", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Log configuration
    config_str = (f"Cross-Entropy Classification (Reconstructed CXR):\n"
                 f"  model_name={args.model_name}, resize_dim={args.resize_dim}\n"
                 f"  process={args.process}, train_source={args.train_source}\n"
                 f"  bsz={args.bsz}, lr={args.lr}, weight_decay={args.weight_decay}\n"
                 f"  epochs={args.epochs}, seed={args.seed}\n"
                 f"  freeze_backbone={args.freeze_backbone}\n"
                 f"  device={device}")

    print(config_str)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(config_str)

    # Setup CSV logging
    csv_cols = ['epoch', 'lr', 'train_loss', 'train_acc', 'train_precision', 'train_recall',
                'train_f1', 'train_auc', 'test_loss', 'test_acc',
                'test_precision', 'test_recall', 'test_f1', 'test_auc',
                'test2_f1', 'test2_auc', 'test3_f1', 'test3_auc']

    if not args.resume:
        with open(os.path.join(log_dir, "results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writeheader()

    # Training loop
    best_f1 = 0.0

    if args.quiet:
        epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress",
                         initial=start_epoch, total=args.epochs)

    for epoch in range(start_epoch, args.epochs):
        # Training
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device)

        # Evaluation
        test_metrics = evaluate_epoch(model, test_dataloader, criterion, device)
        test2_metrics = evaluate_epoch(model, test2_dataloader, criterion, device)
        test3_metrics = evaluate_epoch(model, test3_dataloader, criterion, device)

        # Logging
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {args.lr:.6f}")
            print(f"  Train: Loss={train_metrics['avg_loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, "
                  f"Prec={train_metrics['precision']:.4f}, Rec={train_metrics['recall']:.4f}, "
                  f"F1={train_metrics['f1']:.4f}, AUC={train_metrics['auc']:.4f}")
            print(f"  Test:  Loss={test_metrics['avg_loss']:.4f}, Acc={test_metrics['accuracy']:.4f}, "
                  f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, "
                  f"F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
            print(f"  {external_test_names[0]:8} F1={test2_metrics['f1']:.4f}, AUC={test2_metrics['auc']:.4f}")
            print(f"  {external_test_names[1]:8} F1={test3_metrics['f1']:.4f}, AUC={test3_metrics['auc']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_metrics["avg_loss"]:.4f}',
                'Train_AUC': f'{train_metrics["auc"]:.4f}',
                'Test_F1': f'{test_metrics["f1"]:.4f}',
                'Test_AUC': f'{test_metrics["auc"]:.4f}'
            })
            epoch_pbar.update(1)

        # Save metrics to CSV
        with open(os.path.join(log_dir, "results.csv"), 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writerow({
                'epoch': epoch,
                'lr': args.lr,
                'train_loss': train_metrics['avg_loss'],
                'train_acc': train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_f1': train_metrics['f1'],
                'train_auc': train_metrics['auc'],
                'test_loss': test_metrics['avg_loss'],
                'test_acc': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'test_auc': test_metrics['auc'],
                'test2_f1': test2_metrics['f1'],
                'test2_auc': test2_metrics['auc'],
                'test3_f1': test3_metrics['f1'],
                'test3_auc': test3_metrics['auc']
            })

        # Save checkpoint
        is_best = test_metrics['f1'] > best_f1
        if is_best:
            best_f1 = test_metrics['f1']

        save_checkpoint(
            model, optimizer, epoch, test_metrics, args,
            save_dir=os.path.join(log_dir, "checkpoints"),
            is_best=is_best
        )

    if args.quiet:
        epoch_pbar.close()

    print(f"\nTraining complete! Best F1: {best_f1:.4f}")
    print(f"Results saved to: {log_dir}")


if __name__ == "__main__":
    main()
