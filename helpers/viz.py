import matplotlib.pyplot as plt
import matplotlib


import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import torch

def plot_distance_histograms(normal_distances, nodule_distances, epoch=None, save_path=None):
    """
    Plot histograms of distances for normal vs nodule pairs
    """
    matplotlib.set_loglevel('ERROR')
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(normal_distances, bins=30, alpha=0.7, label=f'Normal pairs (n={len(normal_distances)})', 
             color='blue', density=True)
    plt.hist(nodule_distances, bins=30, alpha=0.7, label=f'Nodule pairs (n={len(nodule_distances)})', 
             color='red', density=True)
    
    # Add mean lines
    normal_mean = np.mean(normal_distances)
    nodule_mean = np.mean(nodule_distances)
    plt.axvline(normal_mean, color='blue', linestyle='--', alpha=0.8, 
                label=f'Normal mean: {normal_mean:.3f}')
    plt.axvline(nodule_mean, color='red', linestyle='--', alpha=0.8, 
                label=f'Nodule mean: {nodule_mean:.3f}')
    
    plt.xlabel('Euclidean Distance Between Lung Pairs')
    plt.ylabel('Density')
    title = f'Distance Distribution Between Left/Right Lung Embeddings'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def generate_tsne_plot(model, dataloader, device, epoch=None, save_path=None):
    """
    Generate t-SNE visualization of embeddings
    """
    matplotlib.set_loglevel('ERROR')
    model.eval()
    
    all_embeddings_l = []
    all_embeddings_r = []
    all_labels = []
    
    with torch.no_grad():
        for _, (img1, img2, labels, _path1, _path2) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            emb1, emb2 = model(img1, img2)
            
            all_embeddings_l.append(emb1.cpu().numpy())
            all_embeddings_r.append(emb2.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings_l = np.vstack(all_embeddings_l)
    embeddings_r = np.vstack(all_embeddings_r)
    labels = np.hstack(all_labels)
    
    # Combine left and right embeddings for t-SNE
    all_embeddings = np.vstack([embeddings_l, embeddings_r])
    all_labels_combined = np.hstack([labels, labels])  # Duplicate labels for L and R
    side_labels = np.hstack([np.zeros(len(labels)), np.ones(len(labels))])  # 0=left, 1=right
    
    # Run t-SNE (subsample if too many points)
    max_samples = 1000
    if len(all_embeddings) > max_samples:
        indices = np.random.choice(len(all_embeddings), max_samples, replace=False)
        all_embeddings = all_embeddings[indices]
        all_labels_combined = all_labels_combined[indices]
        side_labels = side_labels[indices]
    
    # print("Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Color by class (normal/nodule)
    normal_mask = all_labels_combined == 0
    nodule_mask = all_labels_combined == 1
    
    axes[0].scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Normal', s=20)
    axes[0].scatter(embeddings_2d[nodule_mask, 0], embeddings_2d[nodule_mask, 1], 
                   c='red', alpha=0.6, label='Nodule', s=20)
    axes[0].set_title('t-SNE: Colored by Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Color by side (left/right)
    left_mask = side_labels == 0
    right_mask = side_labels == 1
    
    axes[1].scatter(embeddings_2d[left_mask, 0], embeddings_2d[left_mask, 1], 
                   c='green', alpha=0.6, label='Left lung', s=20)
    axes[1].scatter(embeddings_2d[right_mask, 0], embeddings_2d[right_mask, 1], 
                   c='orange', alpha=0.6, label='Right lung', s=20)
    axes[1].set_title('t-SNE: Colored by Side')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if epoch is not None:
        fig.suptitle(f'Embedding Visualization (Epoch {epoch})', fontsize=16)
    else:
        fig.suptitle('Embedding Visualization', fontsize=16)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return embeddings_2d, all_labels_combined, side_labels