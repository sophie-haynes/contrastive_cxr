#!/usr/bin/env python3
"""
Integrated Gradients Visualization for Spatial Siamese Models

This script generates Integrated Gradients attribution maps for paired lung images,
showing which regions of the input contribute most to the distance-based classification.

Usage:
    python integrated_gradients_viz.py \
        --checkpoint path/to/checkpoint.pth \
        --model_name rad \
        --process crop \
        --train_source chestxray14 \
        --use_proj \
        --no_single_embedding \
        --num_samples 20 \
        --steps 50 \
        --output_dir ig_outputs
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


def integrated_gradients_pixel_level(encoder, img_left, img_right, device, steps=50):
    """
    Compute Integrated Gradients at pixel level

    Args:
        encoder: SiameseNetworkWithProjectionSpatial model
        img_left, img_right: Input images (B, C, H, W)
        device: torch device
        steps: Number of interpolation steps

    Returns:
        attributions_left: (H, W) attribution map for left lung
        attributions_right: (H, W) attribution map for right lung
        distance: Final distance value
    """
    encoder.eval()

    # Baseline: black image
    baseline_left = torch.zeros_like(img_left)
    baseline_right = torch.zeros_like(img_right)

    # Generate interpolation coefficients
    alphas = torch.linspace(0, 1, steps).to(device)

    # Storage for gradients
    gradients_left = []
    gradients_right = []

    for alpha in tqdm(alphas, desc="Computing pixel-level IG", leave=False):
        # Interpolate
        interpolated_left = baseline_left + alpha * (img_left - baseline_left)
        interpolated_right = baseline_right + alpha * (img_right - baseline_right)

        # Require gradients
        interpolated_left.requires_grad = True
        interpolated_right.requires_grad = True

        # Forward pass to get embeddings
        emb_left, emb_right = encoder(
            interpolated_left,
            interpolated_right,
            return_embedding=True
        )

        # Compute distance (what we're attributing)
        cosine_sim = torch.sum(emb_left * emb_right, dim=1)
        distance = 1 - cosine_sim
        distance = distance.sum()  # For batch

        # Backward pass
        distance.backward()

        # Store gradients
        gradients_left.append(interpolated_left.grad.detach())
        gradients_right.append(interpolated_right.grad.detach())

        # Zero gradients for next iteration
        encoder.zero_grad()

    # Average gradients across interpolation steps
    avg_gradients_left = torch.stack(gradients_left).mean(dim=0)  # (B, C, H, W)
    avg_gradients_right = torch.stack(gradients_right).mean(dim=0)

    # Integrated gradients = (input - baseline) * avg_gradients
    integrated_grads_left = (img_left - baseline_left) * avg_gradients_left
    integrated_grads_right = (img_right - baseline_right) * avg_gradients_right

    # Aggregate across color channels
    attributions_left = integrated_grads_left.sum(dim=1)  # (B, H, W)
    attributions_right = integrated_grads_right.sum(dim=1)

    # Get final distance for reference
    with torch.no_grad():
        emb_left, emb_right = encoder(img_left, img_right, return_embedding=True)
        cosine_sim = torch.sum(emb_left * emb_right, dim=1)
        final_distance = (1 - cosine_sim).item()

    return (attributions_left[0].cpu().numpy(),
            attributions_right[0].cpu().numpy(),
            final_distance)


def integrated_gradients_spatial_level(encoder, img_left, img_right, device, steps=50):
    """
    Compute Integrated Gradients at spatial feature level (7x7 or 14x14)
    Much faster than pixel-level, shows which spatial regions matter most

    Args:
        encoder: SiameseNetworkWithProjectionSpatial model
        img_left, img_right: Input images (B, C, H, W)
        device: torch device
        steps: Number of interpolation steps

    Returns:
        attr_map_left: (H_spatial, W_spatial) attribution map for left lung
        attr_map_right: (H_spatial, W_spatial) attribution map for right lung
        distance: Final distance value
    """
    encoder.eval()

    # Get spatial features at baseline (black image) and input
    with torch.no_grad():
        # Baseline spatial features
        features_baseline_left = encoder.backbone(torch.zeros_like(img_left))
        spatial_baseline_left = encoder.embedding_head(features_baseline_left)

        features_baseline_right = encoder.backbone(torch.zeros_like(img_right))
        spatial_baseline_right = encoder.embedding_head(features_baseline_right)

        # Input spatial features
        features_left = encoder.backbone(img_left)
        spatial_left = encoder.embedding_head(features_left)

        features_right = encoder.backbone(img_right)
        spatial_right = encoder.embedding_head(features_right)

    # Interpolate at spatial feature level
    alphas = torch.linspace(0, 1, steps).to(device)
    gradients_spatial_left = []
    gradients_spatial_right = []

    for alpha in tqdm(alphas, desc="Computing spatial-level IG", leave=False):
        # Interpolate spatial features
        interp_spatial_left = spatial_baseline_left + alpha * (spatial_left - spatial_baseline_left)
        interp_spatial_right = spatial_baseline_right + alpha * (spatial_right - spatial_baseline_right)

        interp_spatial_left.requires_grad = True
        interp_spatial_right.requires_grad = True

        # Pool and compute distance
        emb_left = F.adaptive_avg_pool2d(interp_spatial_left, 1).flatten(1)
        emb_left = F.normalize(emb_left, dim=1)

        emb_right = F.adaptive_avg_pool2d(interp_spatial_right, 1).flatten(1)
        emb_right = F.normalize(emb_right, dim=1)

        cosine_sim = torch.sum(emb_left * emb_right, dim=1)
        distance = (1 - cosine_sim).sum()

        distance.backward()

        gradients_spatial_left.append(interp_spatial_left.grad.detach())
        gradients_spatial_right.append(interp_spatial_right.grad.detach())

    # Average and integrate
    avg_grads_left = torch.stack(gradients_spatial_left).mean(dim=0)
    avg_grads_right = torch.stack(gradients_spatial_right).mean(dim=0)

    ig_spatial_left = (spatial_left - spatial_baseline_left) * avg_grads_left
    ig_spatial_right = (spatial_right - spatial_baseline_right) * avg_grads_right

    # Aggregate across channels to get spatial maps
    attr_map_left = ig_spatial_left.sum(dim=1)[0]  # (H_spatial, W_spatial)
    attr_map_right = ig_spatial_right.sum(dim=1)[0]

    # Get final distance
    with torch.no_grad():
        emb_left = F.adaptive_avg_pool2d(spatial_left, 1).flatten(1)
        emb_left = F.normalize(emb_left, dim=1)
        emb_right = F.adaptive_avg_pool2d(spatial_right, 1).flatten(1)
        emb_right = F.normalize(emb_right, dim=1)
        cosine_sim = torch.sum(emb_left * emb_right, dim=1)
        final_distance = (1 - cosine_sim).item()

    return (attr_map_left.cpu().numpy(),
            attr_map_right.cpu().numpy(),
            final_distance)


def visualize_integrated_gradients(img_left, img_right,
                                   attr_left, attr_right,
                                   distance, label, name,
                                   save_path=None,
                                   spatial_resolution=None):
    """
    Visualize Integrated Gradients attributions

    Args:
        img_left, img_right: Original images (C, H, W) tensors
        attr_left, attr_right: Attribution maps (H, W) numpy arrays
        distance: Computed distance value
        label: Ground truth label (0=normal, 1=nodule)
        name: Sample name
        save_path: Path to save figure
        spatial_resolution: If provided, title mentions spatial resolution (e.g., "7x7")
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert images to numpy for display
    def to_numpy_img(tensor):
        img = tensor.cpu().numpy()
        if img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        elif img.shape[0] == 1:  # Grayscale
            img = img[0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

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

    # Row 1: Original images and combined attribution
    axes[0, 0].imshow(img_left_np, cmap='gray' if len(img_left_np.shape) == 2 else None)
    axes[0, 0].set_title('Left Lung', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_right_np, cmap='gray' if len(img_right_np.shape) == 2 else None)
    axes[0, 1].set_title('Right Lung', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Combined attribution (absolute difference)
    combined_attr = np.abs(attr_left_upsampled - attr_right_upsampled)
    im1 = axes[0, 2].imshow(combined_attr, cmap='hot', interpolation='bilinear')
    label_text = "Normal" if label == 0 else "Nodule"
    resolution_text = f" ({spatial_resolution})" if spatial_resolution else ""
    axes[0, 2].set_title(f'Combined Attribution{resolution_text}\nDistance: {distance:.3f} | GT: {label_text}',
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)

    # Row 2: Individual attributions with overlays
    # Use symmetric color scale for individual attributions
    vmax = max(np.abs(attr_left_upsampled).max(), np.abs(attr_right_upsampled).max())

    im2 = axes[1, 0].imshow(attr_left_upsampled, cmap='seismic',
                            vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axes[1, 0].set_title('Left Attribution\n(Red=↑distance, Blue=↓distance)', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(attr_right_upsampled, cmap='seismic',
                            vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axes[1, 1].set_title('Right Attribution\n(Red=↑distance, Blue=↓distance)', fontsize=11)
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

    plt.suptitle(f'Integrated Gradients Attribution Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def generate_ig_visualizations(encoder, dataloader, device, args):
    """
    Generate IG visualizations for multiple samples

    Args:
        encoder: Trained spatial siamese model
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

    encoder.eval()

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
        # Fall back to original behavior
        num_nodule_target = None
        num_normal_target = None
        total_target = args.num_samples

    # Track how many of each class we've processed
    samples_counts = {
        'normal': 0,
        'nodule': 0
    }

    # Track statistics
    stats = {
        'normal': {'distances': [], 'attr_magnitudes': []},
        'nodule': {'distances': [], 'attr_magnitudes': []}
    }

    print(f"\nGenerating Integrated Gradients visualizations...")
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

    # First pass: collect all samples by class
    all_samples = {'normal': [], 'nodule': []}

    print("Collecting samples...")
    for batch_idx, batch in enumerate(dataloader):
        img_left, img_right, labels, names, _ = batch

        for i in range(len(img_left)):
            # Get pair name from path
            pair_name = Path(names[i]).parent.name

            # Skip if specific pairs requested and this isn't one of them
            if specific_pair_names and pair_name not in specific_pair_names:
                continue

            label = labels[i].item()
            label_name = 'normal' if label == 0 else 'nodule'
            name = names[i]

            all_samples[label_name].append({
                'img_left': img_left[i],
                'img_right': img_right[i],
                'label': label,
                'name': name,
                'pair_name': pair_name
            })

    print(f"Found {len(all_samples['normal'])} normal and {len(all_samples['nodule'])} nodule samples")

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
        # Select specific numbers
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
        label = sample['label']
        name = sample['name']

        # Move to device
        img_l = img_left.unsqueeze(0).to(device)
        img_r = img_right.unsqueeze(0).to(device)

        # Extract just the filename from the path
        name_clean = Path(name).stem
        parent_dir = Path(name).parent.name
        sample_id = f"{parent_dir}_{name_clean}"

        label_text = "normal" if label == 0 else "nodule"
        print(f"\n[{sample_idx+1}/{len(selected_samples)}] Processing: {sample_id} (label={label_text})")

        # Continue with existing visualization code...
        # Spatial-level IG (faster)
        if 'spatial' in args.methods:
            print("  Computing spatial-level IG...")
            attr_l_spatial, attr_r_spatial, distance_spatial = integrated_gradients_spatial_level(
                encoder, img_l, img_r, device, steps=args.steps
            )

            # Get spatial resolution
            spatial_res = f"{attr_l_spatial.shape[0]}x{attr_l_spatial.shape[1]}"

            # Visualize
            save_path = spatial_dir / f'{sample_id}_spatial_ig.png'
            visualize_integrated_gradients(
                img_left, img_right,
                attr_l_spatial, attr_r_spatial,
                distance_spatial, label, sample_id,
                save_path=save_path,
                spatial_resolution=spatial_res
            )
            plt.close()

            # Track stats
            label_name = 'normal' if label == 0 else 'nodule'
            stats[label_name]['distances'].append(distance_spatial)
            stats[label_name]['attr_magnitudes'].append(
                (np.abs(attr_l_spatial).mean() + np.abs(attr_r_spatial).mean()) / 2
            )

            print(f"    Distance: {distance_spatial:.4f}")
            print(f"    Spatial resolution: {spatial_res}")

        # Pixel-level IG (slower, more precise)
        if 'pixel' in args.methods:
            print("  Computing pixel-level IG...")
            attr_l_pixel, attr_r_pixel, distance_pixel = integrated_gradients_pixel_level(
                encoder, img_l, img_r, device, steps=args.steps
            )

            # Visualize
            save_path = pixel_dir / f'{sample_id}_pixel_ig.png'
            visualize_integrated_gradients(
                img_left, img_right,
                attr_l_pixel, attr_r_pixel,
                distance_pixel, label, sample_id,
                save_path=save_path
            )
            plt.close()

            print(f"    Distance: {distance_pixel:.4f}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("-"*80)
    for label_name in ['normal', 'nodule']:
        if stats[label_name]['distances']:
            avg_dist = np.mean(stats[label_name]['distances'])
            avg_attr = np.mean(stats[label_name]['attr_magnitudes'])
            print(f"{label_name.capitalize()} pairs (n={len(stats[label_name]['distances'])}):")
            print(f"  Avg distance: {avg_dist:.4f}")
            print(f"  Avg attribution magnitude: {avg_attr:.4f}")
    print("="*80)


def OLD_generate_ig_visualizations(encoder, dataloader, device, args):
    """OLD VERSION - keeping for reference"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories for different methods
    pixel_dir = output_dir / "pixel_level"
    spatial_dir = output_dir / "spatial_level"
    pixel_dir.mkdir(exist_ok=True)
    spatial_dir.mkdir(exist_ok=True)

    encoder.eval()
    samples_processed = 0

    # Track statistics
    stats = {
        'normal': {'distances': [], 'attr_magnitudes': []},
        'nodule': {'distances': [], 'attr_magnitudes': []}
    }

    print(f"\nGenerating Integrated Gradients visualizations...")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"IG steps: {args.steps}")
    print(f"Samples to process: {args.num_samples}")
    print("="*80)

    for batch_idx, batch in enumerate(dataloader):
        img_left, img_right, labels, names, _ = batch

        for i in range(len(img_left)):
            if samples_processed >= args.num_samples:
                # Print summary statistics
                print("\n" + "="*80)
                print("Summary Statistics:")
                print("-"*80)
                for label_name in ['normal', 'nodule']:
                    if stats[label_name]['distances']:
                        avg_dist = np.mean(stats[label_name]['distances'])
                        avg_attr = np.mean(stats[label_name]['attr_magnitudes'])
                        print(f"{label_name.capitalize()} pairs:")
                        print(f"  Avg distance: {avg_dist:.4f}")
                        print(f"  Avg attribution magnitude: {avg_attr:.4f}")
                print("="*80)
                return

            # Single sample
            img_l = img_left[i:i+1].to(device)
            img_r = img_right[i:i+1].to(device)
            label = labels[i].item()
            name = names[i]

            # Extract just the filename from the path (remove directory structure)
            # name might be like "split_node21_sets/crop/chestxray14/test/normal/c3222/lung_l.png"
            name_clean = Path(name).stem  # Gets filename without extension
            # Also get the parent directory name for better identification
            parent_dir = Path(name).parent.name
            sample_id = f"{parent_dir}_{name_clean}"

            label_text = "normal" if label == 0 else "nodule"
            print(f"\n[{samples_processed+1}/{args.num_samples}] Processing: {sample_id} (label={label_text})")

            # Spatial-level IG (faster)
            if 'spatial' in args.methods:
                print("  Computing spatial-level IG...")
                attr_l_spatial, attr_r_spatial, distance_spatial = integrated_gradients_spatial_level(
                    encoder, img_l, img_r, device, steps=args.steps
                )

                # Get spatial resolution
                spatial_res = f"{attr_l_spatial.shape[0]}x{attr_l_spatial.shape[1]}"

                # Visualize
                save_path = spatial_dir / f'{sample_id}_spatial_ig.png'
                visualize_integrated_gradients(
                    img_left[i], img_right[i],
                    attr_l_spatial, attr_r_spatial,
                    distance_spatial, label, sample_id,
                    save_path=save_path,
                    spatial_resolution=spatial_res
                )
                plt.close()

                # Track stats
                label_name = 'normal' if label == 0 else 'nodule'
                stats[label_name]['distances'].append(distance_spatial)
                stats[label_name]['attr_magnitudes'].append(
                    (np.abs(attr_l_spatial).mean() + np.abs(attr_r_spatial).mean()) / 2
                )

                print(f"    Distance: {distance_spatial:.4f}")
                print(f"    Spatial resolution: {spatial_res}")

            # Pixel-level IG (slower, more precise)
            if 'pixel' in args.methods:
                print("  Computing pixel-level IG...")
                attr_l_pixel, attr_r_pixel, distance_pixel = integrated_gradients_pixel_level(
                    encoder, img_l, img_r, device, steps=args.steps
                )

                # Visualize
                save_path = pixel_dir / f'{sample_id}_pixel_ig.png'
                visualize_integrated_gradients(
                    img_left[i], img_right[i],
                    attr_l_pixel, attr_r_pixel,
                    distance_pixel, label, sample_id,
                    save_path=save_path
                )
                plt.close()

                print(f"    Distance: {distance_pixel:.4f}")

            samples_processed += 1

    print(f"\n{'='*80}")
    print(f"Generated {samples_processed} visualizations in {output_dir}/")
    print(f"{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Integrated Gradients visualizations for spatial Siamese models"
    )

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to visualize")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["rgb", "grey", "single", "rad", "randinit"])
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--use_proj", action="store_true",
                       help="Set if checkpoint uses projection head")
    parser.add_argument("--no_single_embedding", action="store_true",
                       help="Set if checkpoint uses spatial models")
    parser.add_argument("--early_truncation", action="store_true",
                       help="Set if checkpoint uses early truncation (layer3)")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="../split_node21_sets")
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
    parser.add_argument("--output_dir", type=str, default="ig_outputs",
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
    print(f"Integrated Gradients Visualization")
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

    # Load dataset
    dataset_path = os.path.join(args.dataset_path, args.process, args.train_source, args.test_set)
    print(f"Loading data from: {dataset_path}")

    dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=dataset_path,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=True,
        class_to_idx={'nodule': 1, 'normal': 0},
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
        raise ValueError("This script is designed for spatial models. Use --no_single_embedding flag.")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    encoder.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Generate visualizations
    generate_ig_visualizations(encoder, dataloader, device, args)

    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
