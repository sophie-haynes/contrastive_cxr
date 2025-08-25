#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(1, '../')
import helpers

import torch
from torchvision import transforms


def build_transforms(resize_dim: int):
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Transforms
    base_transform, augment_transform = build_transforms(args.resize_dim)

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
    )

    # test_dataloader = helpers.dataloading.load_image_pair_dataset(
    #     dataset_path=test_path,
    #     batch_size=args.bsz,
    #     crop_size=args.resize_dim,
    #     symmetrical_transforms=args.symmetrical_transforms,
    #     class_to_idx={'nodule': 1, 'normal': 0},
    #     transform=base_transform,
    #     cache_in_ram=args.cache_in_ram,
    # )
    # (test_dataloader is created for parity with your script; not used below.)

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


if __name__ == "__main__":
    main()
