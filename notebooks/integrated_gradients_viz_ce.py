#!/usr/bin/env python3
"""
Integrated Gradients Visualization for Cross-Entropy Classification Models

This script generates Integrated Gradients attribution maps for individual lung images,
showing which regions contribute most to the classification decision (normal vs nodule).

Usage:
    python integrated_gradients_viz_ce.py \
        --checkpoint path/to/checkpoint.pth \
        --model_name rad \
        --process crop \
        --train_source chestxray14 \
        --no_single_embedding \
        --num_samples 20 \
        --steps 50 \
        --output_dir ig_outputs_ce
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(1, '../')
import helpers

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


def integrated_gradients_pixel_level_classification(model, img, target_class, device, steps=50):
    """
    Compute Integrated Gradients at pixel level for classification

    Args:
        model: Siamese network model with classification head
        img: Input image (B, C, H, W)
        target_class: Class to compute attribution for (0=normal, 1=nodule)
        device: torch device
        steps: Number of interpolation steps

    Returns:
        attributions: (H, W) attribution map
        logits: Final predicted logits
    """
    model.eval()

    # Baseline: black image
    baseline = torch.zeros_like(img)

    # Generate interpolation coefficients
    alphas = torch.linspace(0, 1, steps).to(device)

    # Storage for gradients
    gradients = []

    for alpha in tqdm(alphas, desc="Computing pixel-level IG", leave=False):
        # Interpolate
        interpolated = baseline + alpha * (img - baseline)
        interpolated.requires_grad = True

        # Forward pass to get logits (we only need logits for one lung)
        # For Siamese models, we pass the same image twice but only use one output
        logits1, _ = model(interpolated, interpolated, return_logits=True)

        # Get the target class logit
        target_logit = logits1[:, target_class]
        target_score = target_logit.sum()

        # Backward pass
        target_score.backward()

        # Store gradients
        gradients.append(interpolated.grad.detach())

        # Zero gradients for next iteration
        model.zero_grad()

    # Average gradients across interpolation steps
    avg_gradients = torch.stack(gradients).mean(dim=0)  # (B, C, H, W)

    # Integrated gradients = (input - baseline) * avg_gradients
    integrated_grads = (img - baseline) * avg_gradients

    # Aggregate across color channels
    attributions = integrated_grads.sum(dim=1)  # (B, H, W)

    # Get final logits for reference
    with torch.no_grad():
        logits1, _ = model(img, img, return_logits=True)
        final_logits = logits1[0].cpu().numpy()

    return attributions[0].cpu().numpy(), final_logits


def integrated_gradients_spatial_level_classification(model, img, target_class, device, steps=50):
    """
    Compute Integrated Gradients at spatial feature level for classification
    Much faster than pixel-level

    Args:
        model: Siamese network model with classification head
        img: Input image (B, C, H, W)
        target_class: Class to compute attribution for (0=normal, 1=nodule)
        device: torch device
        steps: Number of interpolation steps

    Returns:
        attr_map: (H_spatial, W_spatial) attribution map
        logits: Final predicted logits
    """
    model.eval()

    # Get spatial features at baseline (black image) and input
    with torch.no_grad():
        # Baseline spatial features
        features_baseline = model.backbone(torch.zeros_like(img))
        spatial_baseline = model.embedding_head(features_baseline)

        # Input spatial features
        features = model.backbone(img)
        spatial = model.embedding_head(features)

    # Interpolate at spatial feature level
    alphas = torch.linspace(0, 1, steps).to(device)
    gradients_spatial = []

    for alpha in tqdm(alphas, desc="Computing spatial-level IG", leave=False):
        # Interpolate spatial features
        interp_spatial = spatial_baseline + alpha * (spatial - spatial_baseline)
        interp_spatial.requires_grad = True

        # Pool to get embedding
        embedding = F.adaptive_avg_pool2d(interp_spatial, 1).flatten(1)

        # Pass through classification head
        logits = model.classification_head(embedding)

        # Get target class logit
        target_logit = logits[:, target_class]
        target_score = target_logit.sum()

        target_score.backward()

        gradients_spatial.append(interp_spatial.grad.detach())

    # Average and integrate
    avg_grads = torch.stack(gradients_spatial).mean(dim=0)
    ig_spatial = (spatial - spatial_baseline) * avg_grads

    # Aggregate across channels to get spatial map
    attr_map = ig_spatial.sum(dim=1)[0]  # (H_spatial, W_spatial)

    # Get final logits
    with torch.no_grad():
        embedding = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        final_logits = model.classification_head(embedding)[0].cpu().numpy()

    return attr_map.cpu().numpy(), final_logits


def visualize_integrated_gradients_classification(img_left, img_right,
                                                  attr_left, attr_right,
                                                  logits_left, logits_right,
                                                  label_left, label_right,
                                                  name,
                                                  save_path=None,
                                                  spatial_resolution=None):
    """
    Visualize Integrated Gradients attributions for classification (matching paired format)

    Args:
        img_left, img_right: Original images (C, H, W) tensors
        attr_left, attr_right: Attribution maps (H, W) numpy arrays
        logits_left, logits_right: Predicted logits [normal_logit, nodule_logit]
        label_left, label_right: Ground truth labels (0=normal, 1=nodule)
        name: Sample name
        save_path: Path to save figure
        spatial_resolution: If provided, title mentions spatial resolution (e.g., "7x7")
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert image to numpy for display
    def to_numpy_img(tensor):
        img_np = tensor.cpu().numpy()
        if img_np.shape[0] == 3:  # RGB
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        elif img_np.shape[0] == 1:  # Grayscale
            img_np = img_np[0]
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        return img_np

    img_left_np = to_numpy_img(img_left)
    img_right_np = to_numpy_img(img_right)

    # Upsample attribution maps to image size if needed
    if attr_left.shape != img_left_np.shape[:2]:
        attr_left_upsampled = F.interpolate(
            torch.from_numpy(attr_left).unsqueeze(0).unsqueeze(0).float(),
            size=img_left_np.shape[:2],
            mode='bilinear',
            align_corners=False
        )[0, 0].numpy()

        attr_right_upsampled = F.interpolate(
            torch.from_numpy(attr_right).unsqueeze(0).unsqueeze(0).float(),
            size=img_right_np.shape[:2],
            mode='bilinear',
            align_corners=False
        )[0, 0].numpy()
    else:
        attr_left_upsampled = attr_left
        attr_right_upsampled = attr_right

    # Compute predictions
    probs_left = F.softmax(torch.from_numpy(logits_left), dim=0).numpy()
    probs_right = F.softmax(torch.from_numpy(logits_right), dim=0).numpy()

    pred_left = np.argmax(logits_left)
    pred_right = np.argmax(logits_right)

    pred_label_left = "Normal" if pred_left == 0 else "Nodule"
    pred_label_right = "Normal" if pred_right == 0 else "Nodule"

    true_label_left = "Normal" if label_left == 0 else "Nodule"
    true_label_right = "Normal" if label_right == 0 else "Nodule"

    conf_left = probs_left[pred_left] * 100
    conf_right = probs_right[pred_right] * 100

    # Row 1: Original images and combined classification info
    axes[0, 0].imshow(img_left_np, cmap='gray' if len(img_left_np.shape) == 2 else None)
    axes[0, 0].set_title(f'Left Lung\nGT: {true_label_left} | Pred: {pred_label_left}',
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_right_np, cmap='gray' if len(img_right_np.shape) == 2 else None)
    axes[0, 1].set_title(f'Right Lung\nGT: {true_label_right} | Pred: {pred_label_right}',
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Combined attribution (absolute difference)
    combined_attr = np.abs(attr_left_upsampled - attr_right_upsampled)
    im1 = axes[0, 2].imshow(combined_attr, cmap='hot', interpolation='bilinear')
    resolution_text = f" ({spatial_resolution})" if spatial_resolution else ""
    axes[0, 2].set_title(f'Combined Attribution{resolution_text}\nLeft: {conf_left:.1f}% | Right: {conf_right:.1f}%',
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)

    # Row 2: Individual attributions with overlays
    # Use symmetric color scale for individual attributions
    vmax = max(np.abs(attr_left_upsampled).max(), np.abs(attr_right_upsampled).max())

    im2 = axes[1, 0].imshow(attr_left_upsampled, cmap='seismic',
                            vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axes[1, 0].set_title('Left Attribution\n(Red=↑nodule, Blue=↓nodule)', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(attr_right_upsampled, cmap='seismic',
                            vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axes[1, 1].set_title('Right Attribution\n(Red=↑nodule, Blue=↓nodule)', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    # Overlay on side-by-side image
    if len(img_left_np.shape) == 2:  # Grayscale
        combined_img = np.concatenate([img_left_np, img_right_np], axis=1)
        cmap_img = 'gray'
    else:  # RGB
        combined_img = np.concatenate([img_left_np, img_right_np], axis=1)
        cmap_img = None

    axes[1, 2].imshow(combined_img, cmap=cmap_img)

    # Create combined attribution for overlay
    combined_attr_full = np.concatenate([attr_left_upsampled, attr_right_upsampled], axis=1)

    # Normalize for overlay (use absolute values, make positive attributions visible)
    combined_attr_abs = np.abs(combined_attr_full)
    combined_attr_norm = (combined_attr_abs - combined_attr_abs.min()) / (combined_attr_abs.max() - combined_attr_abs.min() + 1e-8)

    overlay = axes[1, 2].imshow(combined_attr_norm, cmap='hot', alpha=0.4,
                                interpolation='bilinear')
    axes[1, 2].set_title(f'Overlay Heatmap\nSample: {name}', fontsize=11)
    axes[1, 2].axis('off')

    plt.suptitle(f'Integrated Gradients Attribution Analysis - Classification',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def generate_ig_visualizations(model, dataloader, device, args):
    """
    Generate IG visualizations for multiple samples

    Args:
        model: Trained classification model
        dataloader: Test dataloader
        device: torch device
        args: Command line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories for different methods
    pixel_dir = output_dir / "pixel_level"
    spatial_dir = output_dir / "spatial_level"
    pixel_dir.mkdir(exist_ok=True)
    spatial_dir.mkdir(exist_ok=True)

    model.eval()

    # Load specific pairs if provided
    specific_pair_names = None
    if args.specific_pairs:
        specific_pair_names = set(args.specific_pairs)
        print(f"Using specific pairs: {specific_pair_names}")
    elif args.pairs_file:
        with open(args.pairs_file, 'r') as f:
            specific_pair_names = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(specific_pair_names)} pair names from {args.pairs_file}")

    # Determine how many samples of each class to collect
    if specific_pair_names:
        # Specific pairs mode - ignore num_nodule/num_normal
        num_nodule_target = None
        num_normal_target = None
        total_target = len(specific_pair_names)
    elif args.num_nodule is not None and args.num_normal is not None:
        num_nodule_target = args.num_nodule
        num_normal_target = args.num_normal
        total_target = num_nodule_target + num_normal_target
    else:
        num_nodule_target = None
        num_normal_target = None
        total_target = args.num_samples

    # Track statistics
    stats = {
        'normal': {'correct': 0, 'total': 0, 'confidences': []},
        'nodule': {'correct': 0, 'total': 0, 'confidences': []}
    }

    print(f"\nGenerating Integrated Gradients visualizations for CE model...")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"IG steps: {args.steps}")
    if specific_pair_names:
        print(f"Processing {len(specific_pair_names)} specific pairs")
    elif num_nodule_target is not None:
        print(f"Target: {num_nodule_target} nodule + {num_normal_target} normal = {total_target} total samples")
    else:
        print(f"Samples to process: {total_target}")
    print(f"Random seed: {args.seed}")
    print("="*80)

    # Set seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # First pass: collect all pairs by class
    all_samples = {'normal': [], 'nodule': []}

    print("Collecting samples...")
    for batch_idx, batch in enumerate(dataloader):
        # For individual labels dataset
        img1, img2, label1, label2, pair_class_idx, path1, path2 = batch

        # Store pairs (not individual lungs)
        for i in range(len(img1)):
            # Get pair name from path
            pair_name = Path(path1[i]).parent.name

            # Skip if specific pairs requested and this isn't one of them
            if specific_pair_names and pair_name not in specific_pair_names:
                continue

            # Determine pair class based on whether either lung has nodule
            pair_label = pair_class_idx[i].item()  # 0=normal pair, 1=nodule pair
            label_name = 'normal' if pair_label == 0 else 'nodule'

            all_samples[label_name].append({
                'img_left': img1[i],
                'img_right': img2[i],
                'label_left': label1[i].item(),
                'label_right': label2[i].item(),
                'name': path1[i],  # Use left lung path as pair identifier
                'pair_name': pair_name
            })

    print(f"Found {len(all_samples['normal'])} normal and {len(all_samples['nodule'])} nodule pairs")

    # Select samples
    if specific_pair_names:
        # Use all pairs that were found matching the specific names
        selected_samples = all_samples['normal'] + all_samples['nodule']
        # Warn if some requested pairs weren't found
        found_pairs = set(s['pair_name'] for s in selected_samples)
        missing_pairs = specific_pair_names - found_pairs
        if missing_pairs:
            print(f"Warning: {len(missing_pairs)} requested pairs not found: {missing_pairs}")
    elif num_nodule_target is not None:
        # Randomly select specific numbers
        selected_normal = random.sample(all_samples['normal'],
                                       min(num_normal_target, len(all_samples['normal'])))
        selected_nodule = random.sample(all_samples['nodule'],
                                       min(num_nodule_target, len(all_samples['nodule'])))
        selected_samples = selected_normal + selected_nodule
        random.shuffle(selected_samples)
    else:
        # Original behavior: just take first N samples
        selected_normal = all_samples['normal'][:total_target]
        selected_nodule = all_samples['nodule'][:total_target]
        selected_samples = selected_normal + selected_nodule
        random.shuffle(selected_samples)

    print(f"Selected {len(selected_samples)} total pairs for visualization")
    print("="*80)

    # Process selected samples
    for sample_idx, sample in enumerate(selected_samples):
        img_left = sample['img_left']
        img_right = sample['img_right']
        label_left = sample['label_left']
        label_right = sample['label_right']
        name = sample['name']

        # Move to device
        img_left_batch = img_left.unsqueeze(0).to(device)
        img_right_batch = img_right.unsqueeze(0).to(device)

        # Extract sample identifier from path
        parent_dir = Path(name).parent.name
        sample_id = parent_dir  # Use pair name (e.g., 'c3222', 'n0526')

        label_text_left = "normal" if label_left == 0 else "nodule"
        label_text_right = "normal" if label_right == 0 else "nodule"
        print(f"\n[{sample_idx+1}/{len(selected_samples)}] Processing: {sample_id} (left={label_text_left}, right={label_text_right})")

        # Attribute to the nodule class (class 1) to see what drives nodule predictions
        target_class = 1

        # Spatial-level IG (faster)
        if 'spatial' in args.methods:
            print("  Computing spatial-level IG for both lungs...")
            attr_left_spatial, logits_left = integrated_gradients_spatial_level_classification(
                model, img_left_batch, target_class, device, steps=args.steps
            )
            attr_right_spatial, logits_right = integrated_gradients_spatial_level_classification(
                model, img_right_batch, target_class, device, steps=args.steps
            )

            # Get spatial resolution
            spatial_res = f"{attr_left_spatial.shape[0]}x{attr_left_spatial.shape[1]}"

            # Visualize
            save_path = spatial_dir / f'{sample_id}_spatial_ig.png'
            visualize_integrated_gradients_classification(
                img_left, img_right,
                attr_left_spatial, attr_right_spatial,
                logits_left, logits_right,
                label_left, label_right,
                sample_id,
                save_path=save_path,
                spatial_resolution=spatial_res
            )
            plt.close()

            # Track stats for both lungs
            pred_left = np.argmax(logits_left)
            pred_right = np.argmax(logits_right)
            probs_left = F.softmax(torch.from_numpy(logits_left), dim=0).numpy()
            probs_right = F.softmax(torch.from_numpy(logits_right), dim=0).numpy()

            for label, predicted, probs in [(label_left, pred_left, probs_left),
                                             (label_right, pred_right, probs_right)]:
                label_name = 'normal' if label == 0 else 'nodule'
                stats[label_name]['total'] += 1
                if predicted == label:
                    stats[label_name]['correct'] += 1
                stats[label_name]['confidences'].append(probs[predicted])

            print(f"    Left - Logits: {logits_left}, Pred: {'Normal' if pred_left == 0 else 'Nodule'}")
            print(f"    Right - Logits: {logits_right}, Pred: {'Normal' if pred_right == 0 else 'Nodule'}")
            print(f"    Spatial resolution: {spatial_res}")

        # Pixel-level IG (slower, more precise)
        if 'pixel' in args.methods:
            print("  Computing pixel-level IG for both lungs...")
            attr_left_pixel, logits_left = integrated_gradients_pixel_level_classification(
                model, img_left_batch, target_class, device, steps=args.steps
            )
            attr_right_pixel, logits_right = integrated_gradients_pixel_level_classification(
                model, img_right_batch, target_class, device, steps=args.steps
            )

            # Visualize
            save_path = pixel_dir / f'{sample_id}_pixel_ig.png'
            visualize_integrated_gradients_classification(
                img_left, img_right,
                attr_left_pixel, attr_right_pixel,
                logits_left, logits_right,
                label_left, label_right,
                sample_id,
                save_path=save_path
            )
            plt.close()

            pred_left = np.argmax(logits_left)
            pred_right = np.argmax(logits_right)
            print(f"    Left - Logits: {logits_left}, Pred: {'Normal' if pred_left == 0 else 'Nodule'}")
            print(f"    Right - Logits: {logits_right}, Pred: {'Normal' if pred_right == 0 else 'Nodule'}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("-"*80)
    for label_name in ['normal', 'nodule']:
        if stats[label_name]['total'] > 0:
            accuracy = 100.0 * stats[label_name]['correct'] / stats[label_name]['total']
            avg_conf = np.mean(stats[label_name]['confidences']) * 100
            print(f"{label_name.capitalize()} samples (n={stats[label_name]['total']}):")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Avg confidence: {avg_conf:.2f}%")
    print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Integrated Gradients visualizations for CE classification models"
    )

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to visualize")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["rgb", "grey", "single", "rad", "randinit"])
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--no_single_embedding", action="store_true",
                       help="Set if checkpoint uses spatial models")
    parser.add_argument("--early_truncation", action="store_true",
                       help="Set if checkpoint uses early truncation (layer3)")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="../split_node21_sets")
    parser.add_argument("--label_csv", type=str, default="../split_node21_sets/only_nodule_half_labels.csv",
                       help="Path to CSV file with individual lung labels")
    parser.add_argument("--process", type=str, required=True,
                       choices=["lung_seg", "crop", "arch_seg"])
    parser.add_argument("--train_source", type=str, required=True,
                       choices=["chestxray14", "jsrt", "padchest"])
    parser.add_argument("--test_set", type=str, default="test",
                       choices=["test", "train"],
                       help="Which split to visualize (default: test)")

    # IG arguments
    parser.add_argument("--methods", type=str, nargs='+',
                       default=["spatial"],
                       choices=["spatial", "pixel"],
                       help="IG methods to use (spatial=fast, pixel=slow but precise)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of interpolation steps for IG (more=accurate but slower)")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Total number of samples to visualize (deprecated if using --num_nodule and --num_normal)")
    parser.add_argument("--num_nodule", type=int, default=None,
                       help="Number of nodule samples to visualize")
    parser.add_argument("--num_normal", type=int, default=None,
                       help="Number of normal samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sample selection")
    parser.add_argument("--specific_pairs", type=str, nargs='+', default=None,
                       help="Specific pair names to visualize (e.g., 'c3222' 'n0526')")
    parser.add_argument("--pairs_file", type=str, default=None,
                       help="Path to text file with one pair name per line")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="ig_outputs_ce",
                       help="Directory to save visualizations")
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--resize_dim", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_in_ram", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Integrated Gradients Visualization - Classification Model")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Process: {args.process}")
    print(f"Train source: {args.train_source}")

    # Build transforms
    base_transform = transforms.Compose([
        transforms.Resize((args.resize_dim, args.resize_dim)),
        transforms.ToTensor(),
    ])

    # Load dataset with individual labels
    dataset_path = os.path.join(args.dataset_path, args.process, args.train_source, args.test_set)
    print(f"Loading data from: {dataset_path}")

    # Use the individual labels dataset
    dataloader = helpers.dataloading.load_image_pair_dataset_with_individual_labels(
        dataset_path=dataset_path,
        label_csv_path=args.label_csv,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=True,
        transform=base_transform,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )

    # Load model
    print(f"\nLoading model...")

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
        model = helpers.models.SiameseNetworkSpatialEarly(
            backbone,
            embedding_dim=args.embedding_dim,
            freeze_backbone=True,
            num_classes=num_classes
        ).to(device)
    elif args.no_single_embedding:
        model = helpers.models.SiameseNetworkSpatial(
            backbone,
            embedding_dim=args.embedding_dim,
            freeze_backbone=True,
            num_classes=num_classes
        ).to(device)
    else:
        model = helpers.models.SiameseNetwork(
            backbone,
            embedding_dim=args.embedding_dim,
            freeze_backbone=True,
            num_classes=num_classes
        ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Generate visualizations
    generate_ig_visualizations(model, dataloader, device, args)

    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
