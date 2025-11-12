#!/usr/bin/env python3
"""
Integrated Gradients Visualization for Reconstructed Cross-Entropy Models

This script generates Integrated Gradients attribution maps for reconstructed full CXR images
(left lung + horizontally-flipped right lung concatenated side-by-side).

Usage:
    python integrated_gradients_viz_ce_reconstructed.py \
        --checkpoint path/to/checkpoint.pth \
        --model_name rad \
        --process crop \
        --train_source chestxray14 \
        --num_samples 20 \
        --steps 50 \
        --output_dir ig_outputs_ce_reconstructed
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(1, '../')
import helpers

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


class ReconstructedCXRDataset(torch.utils.data.Dataset):
    """
    Dataset that reconstructs full CXR images from paired lung data.
    """
    def __init__(self, paired_dataset, transform):
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
            pair_name: Name of the pair (e.g., 'c3222', 'n0526')
        """
        pair_info = self.pairs[idx]

        # Load lung images
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

        # Create new image with combined width
        combined_width = width_l + width_r
        reconstructed = Image.new('RGB', (combined_width, height_l))
        reconstructed.paste(lung_l, (0, 0))
        reconstructed.paste(lung_r_flipped, (width_l, 0))

        # Apply transform
        if self.transform:
            reconstructed = self.transform(reconstructed)

        # Get pair name from path
        pair_name = Path(pair_info['lungl_path']).parent.name

        return reconstructed, pair_info['class_idx'], pair_name


def integrated_gradients_classification(model, img, target_class, device, steps=50):
    """
    Compute Integrated Gradients for reconstructed image classification

    Args:
        model: Classification model (ResNet-50 with FC head)
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

    for alpha in tqdm(alphas, desc="Computing IG", leave=False):
        # Interpolate
        interpolated = baseline + alpha * (img - baseline)
        interpolated.requires_grad = True

        # Forward pass to get logits
        logits = model(interpolated)

        # Get the target class logit
        target_logit = logits[:, target_class]
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
        final_logits = model(img)[0].cpu().numpy()

    return attributions[0].cpu().numpy(), final_logits


def visualize_integrated_gradients_reconstructed(img, attr,
                                                 logits, label, name,
                                                 save_path=None):
    """
    Visualize Integrated Gradients attributions for reconstructed CXR

    Args:
        img: Original reconstructed image (C, H, W) tensor
        attr: Attribution map (H, W) numpy array
        logits: Predicted logits [normal_logit, nodule_logit]
        label: Ground truth label (0=normal, 1=nodule)
        name: Sample name
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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

    img_np = to_numpy_img(img)

    # Upsample attribution map to image size if needed
    if attr.shape != img_np.shape[:2]:
        attr_upsampled = F.interpolate(
            torch.from_numpy(attr).unsqueeze(0).unsqueeze(0).float(),
            size=img_np.shape[:2],
            mode='bilinear',
            align_corners=False
        )[0, 0].numpy()
    else:
        attr_upsampled = attr

    # Compute predictions
    probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
    predicted_class = np.argmax(logits)
    predicted_label = "Normal" if predicted_class == 0 else "Nodule"
    true_label = "Normal" if label == 0 else "Nodule"
    confidence = probs[predicted_class] * 100

    # Plot 1: Original reconstructed image
    axes[0].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
    axes[0].set_title(f'Reconstructed CXR\nGT: {true_label}', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot 2: Attribution map
    im1 = axes[1].imshow(attr_upsampled, cmap='seismic',
                         vmin=-np.abs(attr_upsampled).max(),
                         vmax=np.abs(attr_upsampled).max(),
                         interpolation='bilinear')
    axes[1].set_title(f'Attribution Map\nPred: {predicted_label} ({confidence:.1f}%)',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Plot 3: Overlay
    axes[2].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)

    # Normalize attribution for overlay (use absolute values)
    attr_abs = np.abs(attr_upsampled)
    attr_norm = (attr_abs - attr_abs.min()) / (attr_abs.max() - attr_abs.min() + 1e-8)

    overlay = axes[2].imshow(attr_norm, cmap='hot', alpha=0.4, interpolation='bilinear')
    axes[2].set_title(f'Overlay Heatmap\nSample: {name}', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f'Integrated Gradients - Reconstructed CXR Classification\n' +
                 f'Logits: Normal={logits[0]:.3f}, Nodule={logits[1]:.3f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def generate_ig_visualizations(model, dataloader, device, args):
    """
    Generate IG visualizations for multiple reconstructed samples

    Args:
        model: Trained classification model
        dataloader: Test dataloader (reconstructed dataset)
        device: torch device
        args: Command line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Determine how many samples to collect
    if specific_pair_names:
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

    print(f"\nGenerating Integrated Gradients visualizations for Reconstructed CE model...")
    print(f"Output directory: {output_dir}")
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

    # Collect all samples
    all_samples = {'normal': [], 'nodule': []}

    print("Collecting samples...")
    for batch in dataloader:
        img, label, pair_name = batch

        for i in range(len(img)):
            # Skip if specific pairs requested and this isn't one of them
            if specific_pair_names and pair_name[i] not in specific_pair_names:
                continue

            label_name = 'normal' if label[i].item() == 0 else 'nodule'
            all_samples[label_name].append({
                'img': img[i],
                'label': label[i].item(),
                'pair_name': pair_name[i]
            })

    print(f"Found {len(all_samples['normal'])} normal and {len(all_samples['nodule'])} nodule samples")

    # Select samples
    if specific_pair_names:
        selected_samples = all_samples['normal'] + all_samples['nodule']
        found_pairs = set(s['pair_name'] for s in selected_samples)
        missing_pairs = specific_pair_names - found_pairs
        if missing_pairs:
            print(f"Warning: {len(missing_pairs)} requested pairs not found: {missing_pairs}")
    elif num_nodule_target is not None:
        selected_normal = random.sample(all_samples['normal'],
                                       min(num_normal_target, len(all_samples['normal'])))
        selected_nodule = random.sample(all_samples['nodule'],
                                       min(num_nodule_target, len(all_samples['nodule'])))
        selected_samples = selected_normal + selected_nodule
        random.shuffle(selected_samples)
    else:
        selected_normal = all_samples['normal'][:total_target]
        selected_nodule = all_samples['nodule'][:total_target]
        selected_samples = selected_normal + selected_nodule
        random.shuffle(selected_samples)

    print(f"Selected {len(selected_samples)} total samples for visualization")
    print("="*80)

    # Process selected samples
    for sample_idx, sample in enumerate(selected_samples):
        img = sample['img']
        label = sample['label']
        pair_name = sample['pair_name']

        # Move to device
        img_batch = img.unsqueeze(0).to(device)

        label_text = "normal" if label == 0 else "nodule"
        print(f"\n[{sample_idx+1}/{len(selected_samples)}] Processing: {pair_name} (label={label_text})")

        # Attribute to the nodule class (class 1) to see what drives nodule predictions
        target_class = 1

        print("  Computing Integrated Gradients...")
        attr, logits = integrated_gradients_classification(
            model, img_batch, target_class, device, steps=args.steps
        )

        # Visualize
        save_path = output_dir / f'{pair_name}_ig.png'
        visualize_integrated_gradients_reconstructed(
            img, attr,
            logits, label, pair_name,
            save_path=save_path
        )
        plt.close()

        # Track stats
        label_name = 'normal' if label == 0 else 'nodule'
        predicted = np.argmax(logits)
        probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
        stats[label_name]['total'] += 1
        if predicted == label:
            stats[label_name]['correct'] += 1
        stats[label_name]['confidences'].append(probs[predicted])

        print(f"    Logits: {logits}")
        print(f"    Prediction: {'Normal' if predicted == 0 else 'Nodule'}")

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
        description="Generate Integrated Gradients visualizations for Reconstructed CE models"
    )

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to visualize")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["rgb", "grey", "single", "rad", "randinit"])

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
    parser.add_argument("--output_dir", type=str, default="ig_outputs_ce_reconstructed",
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
    print(f"Integrated Gradients Visualization - Reconstructed CE Model")
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

    # First load paired dataset without transform
    paired_dataloader = helpers.dataloading.load_image_pair_dataset(
        dataset_path=dataset_path,
        batch_size=args.bsz,
        crop_size=args.resize_dim,
        symmetrical_transforms=False,
        class_to_idx={'nodule': 1, 'normal': 0},
        transform=None,  # No transform yet
        cache_in_ram=args.cache_in_ram,
        num_workers=args.workers
    )

    # Get the paired dataset from the dataloader
    paired_dataset = paired_dataloader.dataset

    # Create reconstructed dataset
    reconstructed_dataset = ReconstructedCXRDataset(
        paired_dataset=paired_dataset,
        transform=base_transform
    )

    # Create dataloader for reconstructed images
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        reconstructed_dataset,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=args.workers
    )

    # Load model
    print(f"\nLoading model...")

    model = helpers.models.load_full_model(
        model_name=args.model_name,
        num_classes=2,
        freeze_backbone=False,  # Don't freeze for IG computation
        device=device
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Generate visualizations
    generate_ig_visualizations(model, dataloader, device, args)

    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
