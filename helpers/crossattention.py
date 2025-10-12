import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention module for comparing spatial features between lung pairs.
    Each spatial position in one lung can attend to positions in the other lung.
    """
    def __init__(self, feature_dim=2048, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Learnable projection matrices for Q, K, V
        self.query = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.value = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm([feature_dim])
        self.norm2 = nn.LayerNorm([feature_dim])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1)
        )
        
    def forward(self, feat_L, feat_R):
        """
        Args:
            feat_L: (B, C, H, W) - left lung features
            feat_R: (B, C, H, W) - right lung features
        Returns:
            enhanced_L, enhanced_R: Cross-attended features
        """
        B, C, H, W = feat_L.shape
        
        # Left attends to Right
        enhanced_L = self._cross_attend(feat_L, feat_R)
        # Right attends to Left  
        enhanced_R = self._cross_attend(feat_R, feat_L)
        
        return enhanced_L, enhanced_R
    
    def _cross_attend(self, query_feat, key_value_feat):
        """
        query_feat attends to key_value_feat
        """
        B, C, H, W = query_feat.shape
        
        # Generate Q, K, V
        Q = self.query(query_feat)      # (B, C, H, W)
        K = self.key(key_value_feat)    # (B, C, H, W)
        V = self.value(key_value_feat)  # (B, C, H, W)
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, heads, H*W, head_dim)
        K = K.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, heads, H*W, head_dim)
        V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, heads, H*W, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, heads, H*W, H*W)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, H*W, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(2, 3).contiguous()  # (B, heads, head_dim, H*W)
        attn_output = attn_output.view(B, C, H, W)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Residual connection + LayerNorm
        query_feat_norm = self.norm1(query_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = query_feat_norm + attn_output
        
        # Feed-forward network
        ffn_output = self.ffn(output)
        output_norm = self.norm2(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = output_norm + ffn_output
        
        return output


class CrossAttentionSiamese(nn.Module):
    """
    Siamese network with cross-attention for lung pair comparison.
    Allows spatial features from left and right lungs to interact before pooling.
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, 
                 num_attn_layers=1, num_heads=8, freeze_backbone=False):
        super(CrossAttentionSiamese, self).__init__()
        
        self.backbone = pretrained_backbone
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(feature_dim=2048, num_heads=num_heads)
            for _ in range(num_attn_layers)
        ])
        
        # Projection head (same as original Siamese)
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward_features(self, x):
        """Extract spatial features from backbone"""
        features = self.backbone(x)  # (B, 2048, H, W)
        
        # Ensure spatial dimensions exist
        if features.dim() == 4:
            if features.shape[-1] != 1 or features.shape[-2] != 1:
                # Already has spatial dimensions (e.g., 7x7)
                return features
            else:
                # Has 1x1 spatial, might need upsampling for attention
                # For now, just return as is
                return features
        else:
            # If backbone outputs flat features, we need to handle this
            # This shouldn't happen with truncated ResNet to layer 8
            raise ValueError(f"Expected 4D feature tensor, got shape {features.shape}")
    
    def forward(self, img_L, img_R):
        """
        Forward pass with cross-attention
        Args:
            img_L: (B, C, H, W) - left lung images
            img_R: (B, C, H, W) - right lung images
        Returns:
            emb_L, emb_R: (B, embedding_dim) - normalized embeddings
        """
        # Extract spatial features
        feat_L = self.forward_features(img_L)  # (B, 2048, 7, 7)
        feat_R = self.forward_features(img_R)  # (B, 2048, 7, 7)
        
        # Apply cross-attention layer(s)
        for cross_attn in self.cross_attn_layers:
            feat_L, feat_R = cross_attn(feat_L, feat_R)
        
        # Global average pooling
        feat_L_pooled = F.adaptive_avg_pool2d(feat_L, output_size=1)  # (B, 2048, 1, 1)
        feat_R_pooled = F.adaptive_avg_pool2d(feat_R, output_size=1)  # (B, 2048, 1, 1)
        
        feat_L_pooled = torch.flatten(feat_L_pooled, 1)  # (B, 2048)
        feat_R_pooled = torch.flatten(feat_R_pooled, 1)  # (B, 2048)
        
        # Project to embedding space
        emb_L = self.embedding_head(feat_L_pooled)  # (B, embedding_dim)
        emb_R = self.embedding_head(feat_R_pooled)  # (B, embedding_dim)
        
        # L2 normalize
        emb_L = F.normalize(emb_L, p=2, dim=1)
        emb_R = F.normalize(emb_R, p=2, dim=1)
        
        return emb_L, emb_R
    
    def get_attention_maps(self, img_L, img_R, layer_idx=0):
        """
        Extract attention maps for visualization
        Returns attention weights showing which regions in R are attended to by L
        """
        feat_L = self.forward_features(img_L)
        feat_R = self.forward_features(img_R)
        
        # Get the specified cross-attention layer
        cross_attn = self.cross_attn_layers[layer_idx]
        
        B, C, H, W = feat_L.shape
        
        # Compute attention weights
        Q = cross_attn.query(feat_L)
        K = cross_attn.key(feat_R)
        
        Q = Q.view(B, cross_attn.num_heads, cross_attn.head_dim, H * W).transpose(2, 3)
        K = K.view(B, cross_attn.num_heads, cross_attn.head_dim, H * W).transpose(2, 3)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(cross_attn.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, H*W, H*W)
        
        # Average over heads
        attn_weights = attn_weights.mean(dim=1)  # (B, H*W, H*W)
        
        # Reshape to spatial dimensions
        attn_weights = attn_weights.view(B, H, W, H, W)
        
        return attn_weights
    # In your crossattention.py, the get_attention_maps method computes attention
# but might have issues. Let's add a safer version:

def get_attention_maps_safe(self, img_L, img_R, layer_idx=0):
    """
    Extract attention maps for visualization
    Returns attention weights showing which regions in R are attended to by L
    """
    if layer_idx >= len(self.cross_attn_layers):
        raise ValueError(f"layer_idx {layer_idx} out of range (have {len(self.cross_attn_layers)} layers)")
    
    feat_L = self.forward_features(img_L)
    feat_R = self.forward_features(img_R)
    
    # Get the specified cross-attention layer
    cross_attn = self.cross_attn_layers[layer_idx]
    
    B, C, H, W = feat_L.shape
    
    # Compute attention weights
    Q = cross_attn.query(feat_L)
    K = cross_attn.key(feat_R)
    
    Q = Q.view(B, cross_attn.num_heads, cross_attn.head_dim, H * W).transpose(2, 3)
    K = K.view(B, cross_attn.num_heads, cross_attn.head_dim, H * W).transpose(2, 3)
    
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(cross_attn.head_dim)
    attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, H*W, H*W)
    
    # Check for NaN/Inf before averaging
    if torch.isnan(attn_weights).any():
        print("WARNING: NaN in attention weights before averaging")
    if torch.isinf(attn_weights).any():
        print("WARNING: Inf in attention weights before averaging")
    
    # Average over heads
    attn_weights = attn_weights.mean(dim=1)  # (B, H*W, H*W)
    
    # Reshape to spatial dimensions
    attn_weights = attn_weights.view(B, H, W, H, W)
    
    return attn_weights