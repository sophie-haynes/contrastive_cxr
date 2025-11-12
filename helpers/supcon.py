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
    
    def forward(self, emb1, emb2, labels, label1=None, label2=None):
        """
        Args:
            emb1, emb2: (batch_size, embedding_dim) - normalized embeddings
            labels: (batch_size,) - pair-level labels (0=normal, 1=nodule) for contrastive pair loss
            label1, label2: (batch_size,) - optional individual lung labels for SupCon loss
                           If None, uses pair-level labels for both lungs (backward compatible)

        Returns:
            total_loss: weighted combination of pair and supcon losses
            pair_loss: just the pair component (for logging)
            supcon_loss: just the supcon component (for logging)
        """
        # 1. Compute pair-wise contrastive loss (always uses pair-level labels)
        pair_loss = self.compute_pair_loss(emb1, emb2, labels)

        # 2. Compute SupCon loss across batch
        # Concatenate all embeddings
        all_embeddings = torch.cat([emb1, emb2], dim=0)  # (2*batch_size, embedding_dim)

        # Use individual labels if provided, otherwise fall back to pair labels
        if label1 is not None and label2 is not None:
            # Individual lung labels for SupCon
            all_labels = torch.cat([label1, label2], dim=0)  # (2*batch_size,)
        else:
            # Backward compatible: duplicate pair labels
            all_labels = torch.cat([labels, labels], dim=0)  # (2*batch_size,)

        supcon_loss = self.supcon(all_embeddings, all_labels)

        # 3. Combine losses
        total_loss = self.alpha * pair_loss + self.beta * supcon_loss

        return total_loss, pair_loss, supcon_loss


class HierarchicalLossWithPrototypes(nn.Module):
    """
    Hierarchical loss (Pair + SupCon) with Prototypes

    Combines pairwise contrastive loss with supervised contrastive loss that
    uses class prototypes for better minority class clustering.

    Based on combining HierarchicalLoss with TTC supervised prototypes approach.
    """
    def __init__(self, alpha=0.5, beta=0.5, margin=1.0, distance='euclidean',
                 temperature=0.1, eps_0=0.3, eps_1=0.5, minority_class=1):
        super(HierarchicalLossWithPrototypes, self).__init__()
        self.alpha = alpha  # Weight for pair loss
        self.beta = beta    # Weight for supcon loss

        # Pair-wise contrastive loss components
        self.margin = margin
        self.distance = distance

        # SupCon loss with prototypes
        self.supcon = SupConLossWithPrototypes(
            temperature=temperature,
            eps_0=eps_0,
            eps_1=eps_1,
            minority_class=minority_class
        )

    def set_prototypes(self, prototypes):
        """Pass prototypes to the underlying SupCon loss"""
        self.supcon.set_prototypes(prototypes)

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

    def forward(self, emb1, emb2, labels, label1=None, label2=None):
        """
        Args:
            emb1, emb2: (batch_size, embedding_dim) - normalized embeddings
            labels: (batch_size,) - pair-level labels (0=normal, 1=nodule) for contrastive pair loss
            label1, label2: (batch_size,) - individual lung labels for SupCon loss with prototypes
                           (required for prototype-based SupCon)

        Returns:
            total_loss: weighted combination of pair and supcon losses
            pair_loss: just the pair component (for logging)
            supcon_loss: just the supcon component (for logging)
        """
        # 1. Compute pair-wise contrastive loss (always uses pair-level labels)
        pair_loss = self.compute_pair_loss(emb1, emb2, labels)

        # 2. Compute SupCon loss with prototypes across batch
        # Concatenate all embeddings
        all_embeddings = torch.cat([emb1, emb2], dim=0)  # (2*batch_size, embedding_dim)

        # Use individual labels for SupCon with prototypes
        if label1 is None or label2 is None:
            raise ValueError("HierarchicalLossWithPrototypes requires individual lung labels (label1, label2)")

        all_labels = torch.cat([label1, label2], dim=0)  # (2*batch_size,)
        supcon_loss = self.supcon(all_embeddings, all_labels)

        # 3. Combine losses
        total_loss = self.alpha * pair_loss + self.beta * supcon_loss

        return total_loss, pair_loss, supcon_loss


class SupConLossWithPrototypes(nn.Module):
    """
    Supervised Contrastive Learning Loss with Prototypes

    Based on "A Tale of Two Classes" (Mildenberger et al., CVPR 2025)
    https://github.com/FranziskaMay/TTC

    This loss extends standard SupCon by introducing class prototype vectors that
    act as additional anchor points in the embedding space. Prototypes help create
    tighter, more compact clusters, especially beneficial for minority classes.

    Key features:
    - Class prototypes learned from training embeddings
    - Margin-based conditional pulling to prototypes
    - Asymmetric margins for handling class imbalance

    Args:
        temperature (float): Temperature parameter for contrastive loss (default: 0.1)
        eps_0 (float): Margin for class 0 (majority class). Samples are pulled to
                      prototype_0 only if sim_to_proto_0 <= sim_to_proto_1 + eps_0
                      (default: 0.3)
        eps_1 (float): Margin for class 1 (minority class). Larger values encourage
                      stronger pulling, helping minority class form tighter clusters
                      (default: 0.5)
        minority_class (int): Index of minority class (0 or 1) (default: 1)
        base_temperature (float): Base temperature for loss scaling (default: 0.1)
    """
    def __init__(self, temperature=0.1, eps_0=0.3, eps_1=0.5,
                 minority_class=1, base_temperature=0.1):
        super(SupConLossWithPrototypes, self).__init__()
        self.temperature = temperature
        self.eps_0 = eps_0  # Margin for majority class
        self.eps_1 = eps_1  # Margin for minority class (typically larger)
        self.minority_class = minority_class
        self.base_temperature = base_temperature
        self.prototypes = None  # Will be set via set_prototypes()

    def set_prototypes(self, prototypes):
        """
        Set the prototype vectors for each class

        Args:
            prototypes: Tensor of shape [2, embedding_dim] containing the
                       prototype vectors for class 0 and class 1
        """
        if prototypes.shape[0] != 2:
            raise ValueError(f"Expected 2 prototypes, got {prototypes.shape[0]}")
        self.prototypes = prototypes
        print(f"Prototypes set with shape: {prototypes.shape}")

    def forward(self, features, labels):
        """
        Compute supervised contrastive loss with prototype pulling

        Args:
            features: (N, embedding_dim) - embeddings from the model
            labels: (N,) - class labels for each embedding

        Returns:
            loss: scalar tensor
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Create mask for positive pairs (same label)
        labels_col = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_col, labels_col.T).float().to(device)

        # If prototypes exist, process them
        if self.prototypes is not None:
            # Normalize prototypes
            prototypes_normed = F.normalize(self.prototypes.to(device), dim=1)

            # Concatenate prototypes to features
            features_with_protos = torch.cat([features, prototypes_normed], dim=0)
        else:
            features_with_protos = features

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features_with_protos.T) / self.temperature

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

        # Apply self-comparison mask to positive pairs mask
        mask = mask * logits_mask

        # Extend logits_mask to include prototypes if they exist
        if self.prototypes is not None:
            # Add columns for prototype comparisons
            proto_mask_cols = torch.ones(batch_size, 2).to(device)
            logits_mask = torch.cat([logits_mask, proto_mask_cols], dim=1)

        # Handle prototype-based pulling with margin conditions
        if self.prototypes is not None:
            # Compute similarity to prototypes (before temperature scaling)
            sim_to_protos = torch.matmul(features, prototypes_normed.T)
            sim_to_proto_0 = sim_to_protos[:, 0]
            sim_to_proto_1 = sim_to_protos[:, 1]

            # Create prototype mask based on margin conditions
            proto_mask = torch.zeros(batch_size, 2).to(device)

            # For class 0 samples: pull to prototype_0 only if not already clearly class 0
            class_0_indices = (labels == 0)
            cond_0 = sim_to_proto_0 <= (sim_to_proto_1 + self.eps_0)
            proto_mask[class_0_indices & cond_0, 0] = 1

            # For class 1 samples (minority): pull to prototype_1 with potentially larger margin
            class_1_indices = (labels == 1)
            cond_1 = sim_to_proto_1 <= (sim_to_proto_0 + self.eps_1)
            proto_mask[class_1_indices & cond_1, 1] = 1

            # Append prototype mask to main positive pairs mask
            mask = torch.cat([mask, proto_mask], dim=1)

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Apply temperature scaling if base_temperature is set
        if self.base_temperature > 0.0:
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        else:
            loss = -mean_log_prob_pos

        # Average over batch
        loss = loss.mean()

        return loss


def compute_prototypes_from_embeddings(embeddings, n_iterations=10000, lr=0.0001, verbose=True):
    """
    Compute prototype vectors by finding the direction that minimizes
    cosine similarity to all embeddings.

    This optimization finds the "most different" direction from the data cluster,
    which serves as an ideal separation axis for binary classification.

    Based on TTC implementation (Mildenberger et al., CVPR 2025):
    https://github.com/FranziskaMay/TTC

    Args:
        embeddings: Tensor of shape [N, embedding_dim] containing all training embeddings
        n_iterations: Number of optimization iterations (default: 10000)
        lr: Learning rate for Adam optimizer (default: 0.0001)
        verbose: Print progress every 500 iterations (default: True)

    Returns:
        prototypes: Tensor of shape [2, embedding_dim] containing [prototype_1, -prototype_1]
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    embedding_dim = embeddings.shape[1]

    # Initialize random vector
    vector = nn.Parameter(torch.randn(embedding_dim))
    optimizer = torch.optim.Adam([vector], lr=lr)

    if verbose:
        print(f"Computing prototypes for {embeddings.shape[0]} embeddings...")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Compute cosine similarity between embeddings and current vector
        cos_sim = F.cosine_similarity(
            embeddings,
            vector.unsqueeze(0).expand_as(embeddings),
            dim=1
        )

        # Minimize the sum of cosine similarities
        loss = torch.sum(cos_sim)
        loss.backward()
        optimizer.step()

        # Renormalize vector to unit length
        vector.data = F.normalize(vector.data, dim=0)

        if verbose and iteration % 500 == 0:
            print(f"  Iteration {iteration}/{n_iterations}, Loss: {loss.item():.4f}")

    # For binary classification, return antipodal pair
    prototype_1 = vector.data
    prototype_2 = -prototype_1

    prototypes = torch.stack([prototype_1, prototype_2])

    if verbose:
        print(f"Prototypes computed successfully!")
        print(f"  Prototype 0 norm: {torch.norm(prototypes[0]).item():.4f}")
        print(f"  Prototype 1 norm: {torch.norm(prototypes[1]).item():.4f}")
        print(f"  Dot product: {torch.dot(prototypes[0], prototypes[1]).item():.4f}")

    return prototypes


class SupConLossV2(nn.Module):
    """
    Spatial Supervised Contrastive Learning Loss (V2)

    Applies SupCon loss at each spatial location independently and averages.
    This preserves spatial information during contrastive learning.

    Instead of pooling spatial features (B, C, H, W) to (B, C) before computing
    contrastive loss, this applies contrastive loss at each (h, w) location:
    - Extract features at position (h, w) for all samples: (N, C)
    - Apply SupCon loss at that location
    - Average losses across all spatial locations

    Args:
        temperature: Temperature parameter for contrastive loss (default: 0.1)
    """
    def __init__(self, temperature=0.1):
        super(SupConLossV2, self).__init__()
        self.temperature = temperature
        self.supcon = SupConLoss(temperature=temperature)

    def forward(self, spatial_features, labels):
        """
        Args:
            spatial_features: (N, C, H, W) - spatial embeddings before pooling
            labels: (N,) - class labels for each sample

        Returns:
            loss: scalar tensor - average SupCon loss across all spatial locations
        """
        N, C, H, W = spatial_features.shape
        device = spatial_features.device

        # Apply SupCon at each spatial location
        total_loss = 0.0
        num_locations = H * W

        for h in range(H):
            for w in range(W):
                # Extract features at this spatial location: (N, C)
                features_at_location = spatial_features[:, :, h, w]

                # Apply standard SupCon loss at this location
                location_loss = self.supcon(features_at_location, labels)
                total_loss += location_loss

        # Average across all spatial locations
        avg_loss = total_loss / num_locations

        return avg_loss


class SupConLossWithPrototypesV2(nn.Module):
    """
    Spatial Supervised Contrastive Learning Loss with Prototypes (V2)

    Combines spatial SupCon (applied at each spatial location) with prototype pulling.
    This is the spatial version of SupConLossWithPrototypes.

    Args:
        temperature: Temperature parameter for contrastive loss (default: 0.1)
        eps_0: Margin for class 0 (majority class) (default: 0.3)
        eps_1: Margin for class 1 (minority class) (default: 0.5)
        minority_class: Index of minority class (0 or 1) (default: 1)
        base_temperature: Base temperature for loss scaling (default: 0.1)
    """
    def __init__(self, temperature=0.1, eps_0=0.3, eps_1=0.5,
                 minority_class=1, base_temperature=0.1):
        super(SupConLossWithPrototypesV2, self).__init__()
        self.temperature = temperature
        self.eps_0 = eps_0
        self.eps_1 = eps_1
        self.minority_class = minority_class
        self.base_temperature = base_temperature
        # Use prototype-based SupCon at each spatial location
        self.supcon_proto = SupConLossWithPrototypes(
            temperature=temperature,
            eps_0=eps_0,
            eps_1=eps_1,
            minority_class=minority_class,
            base_temperature=base_temperature
        )

    def set_prototypes(self, prototypes):
        """
        Set the prototype vectors for each class

        Args:
            prototypes: Tensor of shape [2, embedding_dim] containing the
                       prototype vectors for class 0 and class 1
        """
        self.supcon_proto.set_prototypes(prototypes)

    def forward(self, spatial_features, labels):
        """
        Args:
            spatial_features: (N, C, H, W) - spatial embeddings before pooling
            labels: (N,) - class labels for each sample

        Returns:
            loss: scalar tensor - average SupCon loss with prototypes across all spatial locations
        """
        N, C, H, W = spatial_features.shape
        device = spatial_features.device

        # Apply SupCon with prototypes at each spatial location
        total_loss = 0.0
        num_locations = H * W

        for h in range(H):
            for w in range(W):
                # Extract features at this spatial location: (N, C)
                features_at_location = spatial_features[:, :, h, w]

                # Apply SupCon with prototypes at this location
                location_loss = self.supcon_proto(features_at_location, labels)
                total_loss += location_loss

        # Average across all spatial locations
        avg_loss = total_loss / num_locations

        return avg_loss


class HierarchicalLossV2(nn.Module):
    """
    Spatial Hierarchical Loss (V2): Pairwise Contrastive + Spatial SupCon

    Combines:
    1. Pair-level contrastive loss (on pooled embeddings)
    2. Spatial SupCon loss (at each spatial location)

    Args:
        alpha: Weight for pair loss (default: 0.5)
        beta: Weight for supcon loss (default: 0.5)
        margin: Margin for pair contrastive loss (default: 1.0)
        distance: Distance metric - 'euclidean' or 'cosine' (default: 'euclidean')
        temperature: Temperature for SupCon (default: 0.1)
    """
    def __init__(self, alpha=0.5, beta=0.5, margin=1.0,
                 distance='euclidean', temperature=0.1):
        super(HierarchicalLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.distance = distance

        # Spatial SupCon loss
        self.supcon = SupConLossV2(temperature=temperature)

    def compute_pair_loss(self, emb1, emb2, labels):
        """
        Compute pairwise contrastive loss (on pooled embeddings)

        Args:
            emb1, emb2: (batch_size, embedding_dim) - pooled embeddings from left/right lungs
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

    def forward(self, emb1, emb2, labels, spatial1=None, spatial2=None, label1=None, label2=None):
        """
        Args:
            emb1, emb2: (batch_size, embedding_dim) - pooled embeddings for pair loss
            labels: (batch_size,) - pair-level labels (0=normal, 1=nodule) for pair loss
            spatial1, spatial2: (batch_size, C, H, W) - spatial features for SupCon loss
            label1, label2: (batch_size,) - individual lung labels for SupCon loss
                           If None, uses pair-level labels for both lungs

        Returns:
            total_loss: weighted combination of pair and supcon losses
            pair_loss: just the pair component (for logging)
            supcon_loss: just the supcon component (for logging)
        """
        # 1. Compute pair-wise contrastive loss (on pooled embeddings)
        pair_loss = self.compute_pair_loss(emb1, emb2, labels)

        # 2. Compute Spatial SupCon loss across batch
        if spatial1 is None or spatial2 is None:
            raise ValueError("HierarchicalLossV2 requires spatial features (spatial1, spatial2)")

        # Concatenate all spatial features
        all_spatial = torch.cat([spatial1, spatial2], dim=0)  # (2*batch_size, C, H, W)

        # Use individual labels if provided, otherwise fall back to pair labels
        if label1 is not None and label2 is not None:
            all_labels = torch.cat([label1, label2], dim=0)
        else:
            all_labels = torch.cat([labels, labels], dim=0)

        supcon_loss = self.supcon(all_spatial, all_labels)

        # 3. Combine losses
        total_loss = self.alpha * pair_loss + self.beta * supcon_loss

        return total_loss, pair_loss, supcon_loss


class HierarchicalLossWithPrototypesV2(nn.Module):
    """
    Spatial Hierarchical Loss with Prototypes (V2)

    Combines:
    1. Pair-level contrastive loss (on pooled embeddings)
    2. Spatial SupCon with prototypes (at each spatial location)

    Args:
        alpha: Weight for pair loss (default: 0.5)
        beta: Weight for supcon loss (default: 0.5)
        margin: Margin for pair contrastive loss (default: 1.0)
        distance: Distance metric - 'euclidean' or 'cosine' (default: 'euclidean')
        temperature: Temperature for SupCon (default: 0.1)
        eps_0: Margin for class 0 (majority class) (default: 0.3)
        eps_1: Margin for class 1 (minority class) (default: 0.5)
        minority_class: Index of minority class (default: 1)
    """
    def __init__(self, alpha=0.5, beta=0.5, margin=1.0, distance='euclidean',
                 temperature=0.1, eps_0=0.3, eps_1=0.5, minority_class=1):
        super(HierarchicalLossWithPrototypesV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.distance = distance

        # Spatial SupCon loss with prototypes
        self.supcon = SupConLossWithPrototypesV2(
            temperature=temperature,
            eps_0=eps_0,
            eps_1=eps_1,
            minority_class=minority_class
        )

    def set_prototypes(self, prototypes):
        """Pass prototypes to the underlying SupCon loss"""
        self.supcon.set_prototypes(prototypes)

    def compute_pair_loss(self, emb1, emb2, labels):
        """
        Compute pairwise contrastive loss (on pooled embeddings)

        Args:
            emb1, emb2: (batch_size, embedding_dim) - pooled embeddings from left/right lungs
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

    def forward(self, emb1, emb2, labels, spatial1=None, spatial2=None, label1=None, label2=None):
        """
        Args:
            emb1, emb2: (batch_size, embedding_dim) - pooled embeddings for pair loss
            labels: (batch_size,) - pair-level labels (0=normal, 1=nodule) for pair loss
            spatial1, spatial2: (batch_size, C, H, W) - spatial features for SupCon loss
            label1, label2: (batch_size,) - individual lung labels for SupCon with prototypes
                           (required for prototype-based SupCon)

        Returns:
            total_loss: weighted combination of pair and supcon losses
            pair_loss: just the pair component (for logging)
            supcon_loss: just the supcon component (for logging)
        """
        # 1. Compute pair-wise contrastive loss (on pooled embeddings)
        pair_loss = self.compute_pair_loss(emb1, emb2, labels)

        # 2. Compute Spatial SupCon with prototypes across batch
        if spatial1 is None or spatial2 is None:
            raise ValueError("HierarchicalLossWithPrototypesV2 requires spatial features (spatial1, spatial2)")

        if label1 is None or label2 is None:
            raise ValueError("HierarchicalLossWithPrototypesV2 requires individual lung labels (label1, label2)")

        # Concatenate all spatial features
        all_spatial = torch.cat([spatial1, spatial2], dim=0)  # (2*batch_size, C, H, W)
        all_labels = torch.cat([label1, label2], dim=0)

        supcon_loss = self.supcon(all_spatial, all_labels)

        # 3. Combine losses
        total_loss = self.alpha * pair_loss + self.beta * supcon_loss

        return total_loss, pair_loss, supcon_loss
