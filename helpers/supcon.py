import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al. 2020)
    Adapted for lung pair dataset where both lungs from a pair get the same label.
    
    Paper: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (N, embedding_dim) - L2 normalized embeddings
            labels: (N,) - class labels for each embedding
            
        Returns:
            loss: scalar tensor
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features (in case not already normalized)
        features = F.normalize(features, dim=1)
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix: (N, N)
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
        # Handle case where a sample has no positives
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative mean (removed the temperature scaling factor)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class HierarchicalLoss(nn.Module):
    """
    Combined loss: Pairwise Contrastive + SupCon
    
    Balances two objectives:
    1. Pair-level: Healthy pairs close, pathological pairs far
    2. Batch-level: Learn common patterns across samples
    """
    def __init__(self, alpha=0.5, beta=0.5, margin=1.0, 
                 distance='euclidean', temperature=0.1):
        super(HierarchicalLoss, self).__init__()
        self.alpha = alpha  # Weight for pair loss
        self.beta = beta    # Weight for supcon loss
        
        # Pair-wise contrastive loss components
        self.margin = margin
        self.distance = distance
        
        # SupCon loss
        self.supcon = SupConLoss(temperature=temperature)
        
    def compute_pair_loss(self, emb1, emb2, labels):
        """
        Compute pairwise contrastive loss
        Args:
            emb1, emb2: (batch_size, embedding_dim) - embeddings from left/right lungs
            labels: (batch_size,) - 0=normal (should be close), 1=nodule (should be far)
        """
        if self.distance == 'euclidean':
            distance = F.pairwise_distance(emb1, emb2, p=2)
        elif self.distance == 'cosine':
            cosine_sim = torch.sum(emb1 * emb2, dim=1)
            distance = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        
        # 0='normal' should be pulled together, 1='nodule' pushed apart
        positive_pairs = (1 - labels)  # normal pairs
        negative_pairs = labels        # nodule pairs
        
        loss = torch.mean(
            positive_pairs * torch.pow(distance, 2) +
            negative_pairs * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return loss
    
    def forward(self, emb1, emb2, labels):
        """
        Args:
            emb1, emb2: (batch_size, embedding_dim) - normalized embeddings
            labels: (batch_size,) - pair-level labels (0=normal, 1=nodule)
            
        Returns:
            total_loss: weighted combination of pair and supcon losses
            pair_loss: just the pair component (for logging)
            supcon_loss: just the supcon component (for logging)
        """
        # 1. Compute pair-wise contrastive loss
        pair_loss = self.compute_pair_loss(emb1, emb2, labels)
        
        # 2. Compute SupCon loss across batch
        # Concatenate all embeddings and duplicate labels
        all_embeddings = torch.cat([emb1, emb2], dim=0)  # (2*batch_size, embedding_dim)
        all_labels = torch.cat([labels, labels], dim=0)  # (2*batch_size,)
        
        supcon_loss = self.supcon(all_embeddings, all_labels)
        
        # 3. Combine losses
        total_loss = self.alpha * pair_loss + self.beta * supcon_loss
        
        return total_loss, pair_loss, supcon_loss
