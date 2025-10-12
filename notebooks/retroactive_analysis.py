"""
Retroactive Analysis Functions for Jupyter Notebook
Loads saved checkpoints and computes new visualizations/metrics
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm
import pandas as pd

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return checkpoint['epoch'], checkpoint.get('eval_metrics', {})


# ============================================================================
# EMBEDDING COLLECTION
# ============================================================================

def collect_embeddings(model, dataloader, device, max_samples=2000):
    """Collect embeddings and labels from a dataloader"""
    model.eval()
    
    all_embeddings_l = []
    all_embeddings_r = []
    all_labels = []
    
    total_collected = 0
    
    with torch.no_grad():
        for _, (img1, img2, labels, _, _) in enumerate(tqdm(dataloader, desc="Collecting embeddings", leave=False)):
            if total_collected >= max_samples:
                break
                
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1, emb2 = model(img1, img2)
            
            all_embeddings_l.append(emb1.cpu().numpy())
            all_embeddings_r.append(emb2.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            total_collected += len(labels)
    
    embeddings_l = np.vstack(all_embeddings_l)
    embeddings_r = np.vstack(all_embeddings_r)
    labels = np.hstack(all_labels)
    
    # Combine left and right
    all_embeddings = np.vstack([embeddings_l, embeddings_r])
    all_labels_combined = np.hstack([labels, labels])
    
    # Subsample if still too many
    if len(all_embeddings) > max_samples:
        indices = np.random.choice(len(all_embeddings), max_samples, replace=False)
        all_embeddings = all_embeddings[indices]
        all_labels_combined = all_labels_combined[indices]
    
    return all_embeddings, all_labels_combined


# ============================================================================
# EMBEDDING HEALTH METRICS
# ============================================================================

def compute_embedding_health(embeddings):
    """Compute metrics to detect embedding collapse"""
    # 1. Effective rank
    try:
        S = np.linalg.svd(embeddings, compute_uv=False)
        S_normalized = S / (S.sum() + 1e-10)
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
        effective_rank = np.exp(entropy)
    except:
        effective_rank = -1
    
    # 2. Pairwise distance distribution
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    distances = pdist(embeddings[sample_indices])
    distance_mean = np.mean(distances)
    distance_std = np.std(distances)
    
    # 3. Norm variance
    norms = np.linalg.norm(embeddings, axis=1)
    norm_mean = np.mean(norms)
    norm_std = np.std(norms)
    
    return {
        'effective_rank': effective_rank,
        'distance_mean': distance_mean,
        'distance_std': distance_std,
        'norm_mean': norm_mean,
        'norm_std': norm_std
    }


# ============================================================================
# ATTENTION METRICS
# ============================================================================

def compute_attention_entropy(model, dataloader, device, max_batches=10):
    """Compute attention entropy (only for CrossAttentionSiamese)"""
    if not hasattr(model, 'get_attention_maps'):
        return {'attention_entropy': None, 'normalized_entropy': None}
    
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for batch_idx, (img1, img2, labels, _, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            img1, img2 = img1.to(device), img2.to(device)
            
            try:
                attn_weights = model.get_attention_maps(img1, img2, layer_idx=0)
                B, H, W, _, _ = attn_weights.shape
                attn_flat = attn_weights.view(B, H*W, H*W)
                
                attn_probs = attn_flat + 1e-10
                entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1)
                entropies.append(entropy.mean().cpu().item())
            except Exception as e:
                print(f"Warning: Could not compute attention entropy: {e}")
                return {'attention_entropy': None, 'normalized_entropy': None}
    
    if entropies:
        avg_entropy = np.mean(entropies)
        max_entropy = np.log(H * W)
        normalized_entropy = avg_entropy / max_entropy
        
        return {
            'attention_entropy': avg_entropy,
            'normalized_entropy': normalized_entropy
        }
    
    return {'attention_entropy': None, 'normalized_entropy': None}


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_multi_projection_plot(embeddings, labels, epoch, save_path=None, 
                                   title_prefix="", show=True):
    """Generate PCA, UMAP, and t-SNE visualizations side-by-side"""
    projections = {}
    titles = []
    extra_info = {}
    
    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    projections['PCA'] = pca.fit_transform(embeddings)
    titles.append('PCA')
    extra_info['PCA'] = {
        'explained_var': pca.explained_variance_ratio_
    }
    
    # 2. UMAP
    if UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42, 
                              n_neighbors=15, min_dist=0.1)
            projections['UMAP'] = reducer.fit_transform(embeddings)
            titles.append('UMAP')
        except Exception as e:
            print(f"UMAP failed: {e}")
    
    # 3. t-SNE
    try:
        perplexity = min(30, len(embeddings)//4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        projections['t-SNE'] = tsne.fit_transform(embeddings)
        titles.append('t-SNE')
    except Exception as e:
        print(f"t-SNE failed: {e}")
    
    # Plot
    n_plots = len(projections)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    for ax, method in zip(axes, titles):
        proj = projections[method]
        normal_mask = labels == 0
        nodule_mask = labels == 1
        
        ax.scatter(proj[normal_mask, 0], proj[normal_mask, 1], 
                  c='blue', alpha=0.6, label='Normal', s=20)
        ax.scatter(proj[nodule_mask, 0], proj[nodule_mask, 1], 
                  c='red', alpha=0.6, label='Nodule', s=20)
        ax.set_title(f'{method} Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if method == 'PCA':
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    
    fig.suptitle(f'{title_prefix}Embedding Projections (Epoch {epoch})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close('all')
    
    return projections, extra_info


def visualize_attention_maps(model, dataloader, device, save_path=None, show=True):
    """Visualize attention maps (only for CrossAttentionSiamese)"""
    if not hasattr(model, 'get_attention_maps'):
        print("Model doesn't have attention maps")
        return
    
    model.eval()
    
    for img1, img2, labels, _, _ in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        
        try:
            with torch.no_grad():
                attn_weights = model.get_attention_maps(img1[:1], img2[:1], layer_idx=0)
            
            attn_weights = attn_weights.squeeze(0).cpu().numpy()
            H, W = attn_weights.shape[:2]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            positions = [(0, 0), (0, W-1), (H-1, 0), (H-1, W-1)]
            
            for ax, (h, w) in zip(axes.flat, positions):
                attn_map = attn_weights[h, w].reshape(H, W)
                im = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
                ax.set_title(f'Attention from L[{h},{w}] to all R positions')
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close('all')
            
            break
        except Exception as e:
            print(f"Could not visualize attention: {e}")
            break


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_single_checkpoint(checkpoint_path, model, dataloaders_dict, device, 
                             output_dir=None, show_plots=True):
    """
    Analyze a single checkpoint across multiple datasets
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance (will be modified in-place)
        dataloaders_dict: Dict of {dataset_name: dataloader}
        device: torch device
        output_dir: Where to save outputs (None = don't save)
        show_plots: Whether to display plots in notebook
    
    Returns:
        results_dict: {dataset_name: metrics_dict}
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Load checkpoint
    epoch, old_metrics = load_checkpoint(checkpoint_path, model, device)
    print(f"Epoch: {epoch}")
    
    results = {}
    
    for dataset_name, dataloader in dataloaders_dict.items():
        print(f"\n--- {dataset_name} ---")
        
        # Collect embeddings
        embeddings, labels = collect_embeddings(model, dataloader, device)
        
        # Compute metrics
        health_metrics = compute_embedding_health(embeddings)
        attention_metrics = compute_attention_entropy(model, dataloader, device)
        
        # Generate visualizations
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            proj_path = os.path.join(output_dir, f'{dataset_name}_projections_epoch_{epoch}.png')
            attn_path = os.path.join(output_dir, f'{dataset_name}_attention_epoch_{epoch}.png')
        else:
            proj_path = None
            attn_path = None
        
        generate_multi_projection_plot(embeddings, labels, epoch, 
                                      save_path=proj_path, 
                                      title_prefix=f"{dataset_name} - ",
                                      show=show_plots)
        
        visualize_attention_maps(model, dataloader, device, 
                               save_path=attn_path,
                               show=show_plots)
        
        # Combine metrics
        results[dataset_name] = {
            'epoch': epoch,
            **health_metrics,
            **attention_metrics
        }
        
        # Print summary
        print(f"  Effective Rank: {health_metrics['effective_rank']:.2f}")
        print(f"  Distance Std: {health_metrics['distance_std']:.4f}")
        print(f"  Norm Std: {health_metrics['norm_std']:.4f}")
        if attention_metrics['normalized_entropy'] is not None:
            print(f"  Attention Entropy: {attention_metrics['normalized_entropy']:.3f}")
    
    return results


def analyze_multiple_checkpoints(checkpoint_dir, model, dataloaders_dict, device,
                                output_dir=None, checkpoint_pattern="*.pth",
                                analyze_all=False, show_plots=False):
    """
    Analyze multiple checkpoints across multiple datasets
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model instance
        dataloaders_dict: Dict of {dataset_name: dataloader}
        device: torch device
        output_dir: Where to save outputs
        checkpoint_pattern: Glob pattern for checkpoint files
        analyze_all: If False, only analyze best_model.pth
        show_plots: Whether to display plots (usually False for batch analysis)
    
    Returns:
        DataFrame with all results
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find checkpoints
    if analyze_all:
        checkpoint_files = sorted(checkpoint_dir.glob(checkpoint_pattern))
    else:
        best_model = checkpoint_dir / "best_model.pth"
        if best_model.exists():
            checkpoint_files = [best_model]
        else:
            print("No best_model.pth found, analyzing all...")
            checkpoint_files = sorted(checkpoint_dir.glob(checkpoint_pattern))
    
    print(f"Found {len(checkpoint_files)} checkpoint(s)")
    
    # Analyze each
    all_results = []
    
    for ckpt_path in tqdm(checkpoint_files, desc="Analyzing checkpoints"):
        results = analyze_single_checkpoint(
            str(ckpt_path), model, dataloaders_dict, device,
            output_dir=output_dir, show_plots=show_plots
        )
        
        # Flatten results for DataFrame
        for dataset_name, metrics in results.items():
            row = {
                'checkpoint': ckpt_path.name,
                'dataset': dataset_name,
                **metrics
            }
            all_results.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    if output_dir:
        csv_path = os.path.join(output_dir, 'analysis_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path}")
    
    return df


# ============================================================================
# COMPARISON & VISUALIZATION HELPERS
# ============================================================================

def compare_runs(results_dfs, run_names, metric='effective_rank'):
    """
    Compare a metric across multiple runs
    
    Args:
        results_dfs: List of DataFrames from analyze_multiple_checkpoints
        run_names: List of names for each run
        metric: Metric to compare
    """
    fig, axes = plt.subplots(1, len(results_dfs[0]['dataset'].unique()), 
                            figsize=(6*len(results_dfs[0]['dataset'].unique()), 5))
    
    if len(results_dfs[0]['dataset'].unique()) == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, results_dfs[0]['dataset'].unique()):
        for df, name in zip(results_dfs, run_names):
            subset = df[df['dataset'] == dataset]
            ax.plot(subset['epoch'], subset[metric], marker='o', label=name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_metric_evolution(df, metric='effective_rank', datasets=None):
    """
    Plot how a metric evolves over epochs for multiple datasets
    
    Args:
        df: DataFrame from analyze_multiple_checkpoints
        metric: Metric to plot
        datasets: List of dataset names to include (None = all)
    """
    if datasets is None:
        datasets = df['dataset'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in datasets:
        subset = df[df['dataset'] == dataset].sort_values('epoch')
        ax.plot(subset['epoch'], subset[metric], marker='o', label=dataset)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
