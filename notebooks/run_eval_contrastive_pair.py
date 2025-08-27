#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(1, '../')
import helpers

import torch
from torchvision import transforms


def build_transforms(resize_dim: int,single = False):
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
        return base_transform, augment_transform
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


def train_siamese_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for _, (img1, img2, labels, _path1, _path2) in enumerate(dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)

def eval_siamese_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    normal_distances = []
    nodule_distances = []

    import torch.nn.functional as F
    import numpy as np
    
    with torch.no_grad():
        for _, (img1, img2, labels, _path1, _path2) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Get embeddings
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            total_loss += loss.item()
            
            # Calculate distances between pairs
            distances = F.pairwise_distance(emb1, emb2, p=2)
            
            # Separate by class (0=normal should be close, 1=nodule should be far)
            normal_mask = (labels == 0)
            nodule_mask = (labels == 1)
            
            if normal_mask.sum() > 0:
                normal_distances.extend(distances[normal_mask].cpu().numpy())
            if nodule_mask.sum() > 0:
                nodule_distances.extend(distances[nodule_mask].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / max(len(dataloader), 1)
    normal_mean = np.mean(normal_distances) if normal_distances else 0
    nodule_mean = np.mean(nodule_distances) if nodule_distances else 0
    separation = nodule_mean - normal_mean  # Should be positive for good performance
    
    return {
        'avg_loss': avg_loss,
        'normal_dist_mean': normal_mean,
        'nodule_dist_mean': nodule_mean,
        'separation': separation,
        'num_normal_pairs': len(normal_distances),
        'num_nodule_pairs': len(nodule_distances)
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Siamese network on lung-pair images with contrastive loss."
    )
    parser.add_argument("--model_name", type=str, choices=["rgb", "grey", "single", "rad"],
                        required=True, help="Backbone to load via helpers.models.load_truncated_model")
    parser.add_argument("--resize_dim", type=int, default=224)
    parser.add_argument("--dataset_path", type=str, default="../split_node21")
    parser.add_argument("--process", type=str, choices=["lung_seg", "crop", "arch_seg"],
                        required=True, help="Subfolder inside dataset_path")
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--cache_in_ram", action="store_true", help="Enable RAM caching of images")
    parser.add_argument("--symmetrical_transforms", action="store_true",
                        help="Apply identical random transforms to each pair")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the backbone during training")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--distance", type=str, choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Transforms
    base_transform, augment_transform = build_transforms(args.resize_dim,single= True if (args.model_name == "single") else False)

    # Paths
    train_path = os.path.join(args.dataset_path, args.process, "train")
    test_path = os.path.join(args.dataset_path, args.process, "test")

    # Dataloaders (train uses augmentation, test uses base)
    train_dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=train_path,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        class_to_idx={'nodule': 1, 'normal': 0},
        transform=augment_transform,
        cache_in_ram=args.cache_in_ram,
        single= True if (args.model_name == "single") else False,
        num_workers=args.workers
    )

    test_dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=test_path,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        class_to_idx={'nodule': 1, 'normal': 0},
        transform=base_transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )
    

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, loss, optim
    backbone = helpers.models.load_truncated_model(args.model_name)
    model = helpers.models.SiameseNetwork(
        backbone, embedding_dim=128, freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = helpers.losses.ContrastiveLoss(margin=args.margin, distance=args.distance)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training with config:\n"
          f"  model_name={args.model_name}, resize_dim={args.resize_dim}, process={args.process}\n"
          f"  bsz={args.bsz}, cache_in_ram={args.cache_in_ram}, symmetrical_transforms={args.symmetrical_transforms}\n"
          f"  freeze_backbone={args.freeze_backbone}, margin={args.margin}, distance={args.distance}\n"
          f"  lr={args.lr}, epochs={args.epochs}\n"
          f"  device={device}")

    for epoch in range(args.epochs):
        avg_loss = train_siamese_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

        
        train_metrics = eval_siamese_epoch(model, train_dataloader, criterion, device)
        # print(f"  Eval Loss: {train_metrics['avg_loss']:.4f}")
        print(f"  Train Normal pairs distance: {train_metrics['normal_dist_mean']:.4f}")
        print(f"  Train Nodule pairs distance: {train_metrics['nodule_dist_mean']:.4f}")
        print(f"  Train Separation: {train_metrics['separation']:.4f} {'✅' if train_metrics['separation'] > 0 else '❌'}")
        print(f"  Train Pairs evaluated: {train_metrics['num_normal_pairs']} normal, {train_metrics['num_nodule_pairs']} nodule")

        
        eval_metrics = eval_siamese_epoch(model, test_dataloader, criterion, device)
        print(f"  Test Loss: {eval_metrics['avg_loss']:.4f}")
        print(f"  Test Normal pairs distance: {eval_metrics['normal_dist_mean']:.4f}")
        print(f"  Test Nodule pairs distance: {eval_metrics['nodule_dist_mean']:.4f}")
        print(f"  Test Separation: {eval_metrics['separation']:.4f} {'✅' if eval_metrics['separation'] > 0 else '❌'}")
        print(f"  Test Pairs evaluated: {eval_metrics['num_normal_pairs']} normal, {eval_metrics['num_nodule_pairs']} nodule")


if __name__ == "__main__":
    main()
