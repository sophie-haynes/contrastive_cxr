import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss for siamese networks
    Supports both euclidean (classic) and cosine distance
    Note: L2 Norm may work better with margin = 0.5 and cosine distance
    """
    def __init__(self, margin=1.0, distance='euclidean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        
    def forward(self, emb1, emb2, labels):
        """
        Args:
            emb1, emb2: L2-normalised embeddings from siamese network
            labels: 1 for 'nodule' (no match), 0 for 'normal' (match)
        """
        if self.distance == 'euclidean':
            distance = F.pairwise_distance(emb1, emb2, p=2)
        elif self.distance == 'cosine':
            # Cosine distance = 1 - cosine similarity
            # For L2-normalised vectors: cosine_sim = dot_product
            cosine_sim = torch.sum(emb1 * emb2, dim=1)
            distance = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        
        # Invert labels: 0='normal' should be pulled together, 1='nodule' pushed apart
        # (1-labels) to convert: 0->1 (attract), 1->0 (repel)
        positive_pairs = (1 - labels)  # normal pairs should be close
        negative_pairs = labels        # nodule pairs should be far
        
        loss = torch.mean(
            positive_pairs * torch.pow(distance, 2) +  # Pull normal pairs together
            negative_pairs * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)  # Push nodule pairs apart
        )
        return loss