#!/usr/bin/env python3
import os
import sys
import csv
import argparse

sys.path.insert(1, '../')
import helpers

import torch
from torchvision import transforms
from tqdm import tqdm

# Import the new losses
from helpers.supcon import SupConLoss, HierarchicalLoss


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
        print(f"✅ New best model saved at epoch {epoch}!")
    
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


def train_supcon_epoch(model, dataloader, criterion, optimizer, device, loss_type='supcon'):
    """
    Training loop for SupCon or Hierarchical loss
    
    Args:
        loss_type: 'supcon' or 'hierarchical'
    """
    model.train()
    total_loss = 0.0
    total_pair_loss = 0.0
    total_supcon_loss = 0.0
    
    for _, (img1, img2, labels, _path1, _path2) in enumerate(dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings
        emb1, emb2 = model(img1, img2)
        
        # Compute loss based on type
        if loss_type == 'supcon':
            # Pure SupCon: concatenate embeddings and labels
            all_embeddings = torch.cat([emb1, emb2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            loss = criterion(all_embeddings, all_labels)
            
        elif loss_type == 'hierarchical':
            # Hierarchical: returns (total_loss, pair_loss, supcon_loss)
            loss, pair_loss, supcon_loss = criterion(emb1, emb2, labels)
            total_pair_loss += pair_loss.item()
            total_supcon_loss += supcon_loss.item()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / max(len(dataloader), 1)
    
    if loss_type == 'hierarchical':
        avg_pair_loss = total_pair_loss / max(len(dataloader), 1)
        avg_supcon_loss = total_supcon_loss / max(len(dataloader), 1)
        return avg_loss, avg_pair_loss, avg_supcon_loss
    
    return avg_loss


def eval_supcon_epoch(model, dataloader, criterion, device, loss_type='supcon'):
    """Evaluation loop with distance metrics and embedding quality"""
    model.eval()
    total_loss = 0.0
    total_pair_loss = 0.0
    total_supcon_loss = 0.0
    
    normal_distances = []
    nodule_distances = []
    
    # Collect all embeddings for quality metrics
    all_embeddings_list = []
    all_labels_list = []

    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    with torch.no_grad():
        for _, (img1, img2, labels, _path1, _path2) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Get embeddings
            emb1, emb2 = model(img1, img2)
            
            # Store for quality metrics
            all_embeddings_list.append(emb1.cpu())
            all_embeddings_list.append(emb2.cpu())
            all_labels_list.append(labels.cpu())
            all_labels_list.append(labels.cpu())
            
            # Compute loss
            if loss_type == 'supcon':
                all_embeddings = torch.cat([emb1, emb2], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                loss = criterion(all_embeddings, all_labels)
            elif loss_type == 'hierarchical':
                loss, pair_loss, supcon_loss = criterion(emb1, emb2, labels)
                total_pair_loss += pair_loss.item()
                total_supcon_loss += supcon_loss.item()
            
            total_loss += loss.item()
            
            # Calculate distances between pairs
            distances = F.pairwise_distance(emb1, emb2, p=2)
            
            # Separate by class
            normal_mask = (labels == 0)
            nodule_mask = (labels == 1)
            
            if normal_mask.sum() > 0:
                normal_distances.extend(distances[normal_mask].cpu().numpy())
            if nodule_mask.sum() > 0:
                nodule_distances.extend(distances[nodule_mask].cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings_np = torch.cat(all_embeddings_list, dim=0).numpy()
    all_labels_np = torch.cat(all_labels_list, dim=0).numpy()
    
    # Calculate basic metrics
    avg_loss = total_loss / max(len(dataloader), 1)
    normal_mean = np.mean(normal_distances) if normal_distances else 0
    nodule_mean = np.mean(nodule_distances) if nodule_distances else 0
    separation = nodule_mean - normal_mean
    
    # Calculate embedding quality metrics
    silhouette = silhouette_score(all_embeddings_np, all_labels_np) if len(np.unique(all_labels_np)) > 1 else 0
    davies_bouldin = davies_bouldin_score(all_embeddings_np, all_labels_np) if len(np.unique(all_labels_np)) > 1 else 0
    
    # Embedding std (collapse detection)
    embedding_std = np.std(all_embeddings_np)
    
    metrics = {
        'avg_loss': avg_loss,
        'normal_dist_mean': normal_mean,
        'nodule_dist_mean': nodule_mean,
        'separation': separation,
        'num_normal_pairs': len(normal_distances),
        'num_nodule_pairs': len(nodule_distances),
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'embedding_std': embedding_std
    }
    
    if loss_type == 'hierarchical':
        metrics['pair_loss'] = total_pair_loss / max(len(dataloader), 1)
        metrics['supcon_loss'] = total_supcon_loss / max(len(dataloader), 1)
    
    return metrics


def eval_supcon_epoch_with_viz(model, dataloader, criterion, device, epoch, testset, 
                               plt_path="plots/subsets", loss_type='supcon'):
    """Evaluation with visualization"""
    metrics = eval_supcon_epoch(model, dataloader, criterion, device, loss_type)
    
    import numpy as np
    
    # Collect distances for plotting
    normal_distances = []
    nodule_distances = []
    
    model.eval()
    with torch.no_grad():
        for _, (img1, img2, labels, _path1, _path2) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1, emb2 = model(img1, img2)
            
            import torch.nn.functional as F
            distances = F.pairwise_distance(emb1, emb2, p=2)
            
            normal_mask = (labels == 0)
            nodule_mask = (labels == 1)
            
            if normal_mask.sum() > 0:
                normal_distances.extend(distances[normal_mask].cpu().numpy())
            if nodule_mask.sum() > 0:
                nodule_distances.extend(distances[nodule_mask].cpu().numpy())
    
    # Distance histogram
    helpers.viz.plot_distance_histograms(
        normal_distances, nodule_distances, 
        epoch=epoch, 
        save_path=os.path.join(plt_path, f'{testset}_distances_epoch_{epoch}.png')
    )
    
    # t-SNE plot
    helpers.viz.generate_tsne_plot(
        model, dataloader, device, 
        epoch=epoch, 
        save_path=os.path.join(plt_path, f'{testset}_tsne_epoch_{epoch}.png')
    )
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with SupCon or Hierarchical loss on lung pairs"
    )
    parser.add_argument("--model_name", type=str, 
                       choices=["rgb", "grey", "single", "rad", "randinit"],
                       required=True)
    parser.add_argument("--resize_dim", type=int, default=224)
    parser.add_argument("--dataset_path", type=str, default="../split_node21_sets")
    parser.add_argument("--process", type=str, 
                       choices=["lung_seg", "crop", "arch_seg"], required=True)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--cache_in_ram", action="store_true")
    parser.add_argument("--symmetrical_transforms", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    
    # Loss configuration
    parser.add_argument("--loss_type", type=str, 
                       choices=["supcon", "hierarchical"], 
                       default="supcon",
                       help="Type of loss: pure supcon or hierarchical (pair+supcon)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for SupCon loss")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for pair loss in hierarchical mode")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="Weight for supcon loss in hierarchical mode")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin for pair contrastive loss (hierarchical mode)")
    parser.add_argument("--distance", type=str, 
                       choices=["euclidean", "cosine"], 
                       default="euclidean",
                       help="Distance metric for pair loss (hierarchical mode)")
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plot_freq", type=int, default=5)
    parser.add_argument("--plot_dir", type=str, default="plots/subsets")
    parser.add_argument("--log_dir", type=str, default="logs/subsets")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument('-q', '--quiet', action="store_true")
    parser.add_argument("--train_source", type=str, 
                       choices=["chestxray14", "jsrt", "padchest"],
                       required=True)
    return parser.parse_args()


def main():
    args = parse_args()

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

    # Dataloaders
    train_dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=train_path,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        class_to_idx={'nodule': 1, 'normal': 0},
        transform=augment_transform,
        cache_in_ram=args.cache_in_ram,
        single=(args.model_name == "single"),
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

    test2_dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=test2_path,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=args.symmetrical_transforms,
        class_to_idx={'nodule': 1, 'normal': 0},
        transform=base_transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )

    test3_dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=test3_path,
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

    # Model
    backbone = helpers.models.load_truncated_model(args.model_name)
    model = helpers.models.SiameseNetwork(
        backbone, 
        embedding_dim=128, 
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # Loss function
    if args.loss_type == 'supcon':
        criterion = SupConLoss(temperature=args.temperature)
    elif args.loss_type == 'hierarchical':
        criterion = HierarchicalLoss(
            alpha=args.alpha,
            beta=args.beta,
            margin=args.margin,
            distance=args.distance,
            temperature=args.temperature
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup logging directories
    exp_name = "_".join([
        args.model_name, 
        "frz" if args.freeze_backbone else "unfrz",
        args.loss_type,
        args.process, 
        str(args.run)
    ])
    
    plot_dir = os.path.join(args.plot_dir, args.train_source, exp_name)
    log_dir = os.path.join(args.log_dir, args.train_source, exp_name)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Log configuration
    config_str = (f"Training with SupCon configuration:\n"
                 f"  model_name={args.model_name}, resize_dim={args.resize_dim}\n"
                 f"  loss_type={args.loss_type}, temperature={args.temperature}\n"
                 f"  alpha={args.alpha}, beta={args.beta} (hierarchical only)\n"
                 f"  margin={args.margin}, distance={args.distance}\n"
                 f"  bsz={args.bsz}, lr={args.lr}, epochs={args.epochs}\n"
                 f"  device={device}")
    
    print(config_str)
    
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(config_str)

    # Setup CSV logging
    csv_cols = ['avg_loss', 'normal_dist_mean', 'nodule_dist_mean', 
                'separation', 'num_normal_pairs', 'num_nodule_pairs',
                'silhouette_score', 'davies_bouldin_score', 'embedding_std']
    
    if args.loss_type == 'hierarchical':
        csv_cols.extend(['pair_loss', 'supcon_loss'])
    
    for dataset_name in [f"train-{args.train_source}", 
                        f"test-{args.train_source}",
                        f"test-{external_test_names[0]}",
                        f"test-{external_test_names[1]}"]:
        with open(os.path.join(log_dir, f"{dataset_name}_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writeheader()

    # Training loop
    if args.quiet:
        epoch_pbar = tqdm(range(args.epochs), desc="Training Progress")

    best_separation = -float('inf')

    for epoch in range(args.epochs):
        # Training
        train_result = train_supcon_epoch(
            model, train_dataloader, criterion, optimizer, device, args.loss_type
        )
        
        if args.loss_type == 'hierarchical':
            avg_loss, avg_pair_loss, avg_supcon_loss = train_result
        else:
            avg_loss = train_result
        
        # Evaluation
        if epoch % args.plot_freq == 0:
            train_metrics = eval_supcon_epoch_with_viz(
                model, train_dataloader, criterion, device, epoch, 
                f"train-{args.train_source}", plot_dir, args.loss_type
            )
            eval_metrics = eval_supcon_epoch_with_viz(
                model, test_dataloader, criterion, device, epoch,
                f"test-{args.train_source}", plot_dir, args.loss_type
            )
            eval_metrics2 = eval_supcon_epoch_with_viz(
                model, test2_dataloader, criterion, device, epoch,
                f"test-{external_test_names[0]}", plot_dir, args.loss_type
            )
            eval_metrics3 = eval_supcon_epoch_with_viz(
                model, test3_dataloader, criterion, device, epoch,
                f"test-{external_test_names[1]}", plot_dir, args.loss_type
            )
        else:
            train_metrics = eval_supcon_epoch(
                model, train_dataloader, criterion, device, args.loss_type
            )
            eval_metrics = eval_supcon_epoch(
                model, test_dataloader, criterion, device, args.loss_type
            )
            eval_metrics2 = eval_supcon_epoch(
                model, test2_dataloader, criterion, device, args.loss_type
            )
            eval_metrics3 = eval_supcon_epoch(
                model, test3_dataloader, criterion, device, args.loss_type
            )
        
        # Logging
        if not args.quiet:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            if args.loss_type == 'hierarchical':
                print(f"  Train: Loss={avg_loss:.4f}, Pair={avg_pair_loss:.4f}, SupCon={avg_supcon_loss:.4f}")
            else:
                print(f"  Train: Loss={avg_loss:.4f}")
            print(f"  Train Sep: {train_metrics['separation']:.4f}")
            print(f"  Test Sep: {eval_metrics['separation']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'Train_Loss': f'{avg_loss:.4f}',
                'Test_Sep': f'{eval_metrics["separation"]:.4f}'
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
        
        # Save checkpoint
        is_best = eval_metrics['separation'] > best_separation
        if is_best:
            best_separation = eval_metrics['separation']
        
        save_model_checkpoint(
            model, optimizer, epoch, avg_loss, eval_metrics, args,
            save_dir=os.path.join(log_dir, "checkpoints"),
            is_best=is_best
        ) if epoch % 10 == 0 or is_best else None  # Save every 10 epochs or if best

    if args.quiet:
        epoch_pbar.close()


if __name__ == "__main__":
    main()

