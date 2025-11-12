import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import Conv2d
from torchvision.models import resnet50
from torch.nn import Linear
from torch import load as tload
from torch import device

def get_device():
    from torch import cuda
    return device("cuda" if cuda.is_available() else "cpu")

def convert_to_single_channel(model):
    """
    Modifies the first convolutional layer of a given model to accept single-channel input.

    Args:
        model (torch.nn.Module): The model to be modified.

    Returns:
        torch.nn.Module: The modified model with a single-channel input.
    """
    # Identify the first convolutional layer
    conv1 = None
    for name, layer in model.named_modules():
        if isinstance(layer, Conv2d):
            conv1 = layer
            conv1_name = name
            break

    if conv1 is None:
        raise ValueError("The model does not have a Conv2D layer.")

    # Create a new convolutional layer with the same parameters
    # except for the input channels
    new_conv1 = Conv2d(
        in_channels=1,  # Change input channels to 1
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None
    )

    # Replace the old conv1 layer with the new one
    def recursive_setattr(model, attr, value):
        attr_list = attr.split('.')
        for attr_name in attr_list[:-1]:
            model = getattr(model, attr_name)
        setattr(model, attr_list[-1], value)

    recursive_setattr(model, conv1_name, new_conv1)

    return model

def load_trained_resnet50(model_path, single=False, num_classes=2,device=None):
    """Helper to load model from training for evaluation."""
    model = resnet50(weights=None)
    if single:
        model = convert_to_single_channel(model)
    # get input shape
    num_ftrs = model.fc.in_features
    # add linear classifer
    model.fc = Linear(num_ftrs, num_classes)
    # load model to CPU
    model.load_state_dict(tload(model_path, map_location="cpu")['model'])
    # set to eval mode
    model = model.eval()
    if not device:
        device = get_device()
    # load to device
    model = model.to(device)

    return model

def load_full_model(model_name, num_classes=2, freeze_backbone=False, device=None):
    """
    Load a full ResNet50 model with pretrained backbone and classification head.

    Args:
        model_name: One of 'rgb', 'grey', 'single', 'rad', 'randinit'
        num_classes: Number of output classes (default: 2 for binary classification)
        freeze_backbone: If True, freeze all layers except the final FC layer
        device: Device to load model to (default: auto-detect)

    Returns:
        Complete ResNet50 model with modified FC layer for classification
    """
    if model_name.lower() == "rad":
        # RadImageNet - this is already truncated, so we need to add avgpool + fc
        from helpers.radimagenet import RadImageNetBackbone
        backbone = RadImageNetBackbone()
        backbone.load_state_dict(torch.load("../models/radimagenet_resnet50.pt"))

        # Add avgpool and fc layer to complete the model
        # The truncated backbone outputs (B, 2048, H, W), need to pool and classify
        model = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    elif model_name.lower() == "rgb":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = Linear(num_ftrs, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    elif model_name.lower() == "grey":
        model = resnet50(weights=None)
        model.load_state_dict(torch.load("../models/grey_3c_model_89.pth", map_location='cpu')["model"])
        num_ftrs = model.fc.in_features
        model.fc = Linear(num_ftrs, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    elif model_name.lower() == "single":
        model = resnet50(weights=None)
        model = convert_to_single_channel(model)
        model.load_state_dict(torch.load("../models/grey_1c_model_89.pth", map_location='cpu')["model"])
        num_ftrs = model.fc.in_features
        model.fc = Linear(num_ftrs, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    elif model_name.lower() == "randinit":
        model = resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = Linear(num_ftrs, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
    else:
        raise ValueError(f"Invalid model name: {model_name}. Expected one of: rgb, grey, single, rad, randinit")

    # Move to device
    if not device:
        device = get_device()
    model = model.to(device)

    return model

def load_contrastive_backbone(checkpoint_path, single_embedding=True, device=None):
    """
    Load just the ResNet50 backbone from a trained contrastive model checkpoint.

    This allows you to use contrastive-pretrained backbones as initialization
    for other tasks, similar to how you'd use 'rad' or 'rgb' pretrained weights.

    Args:
        checkpoint_path: Path to the contrastive model checkpoint (.pth file)
        single_embedding: If True, include avgpool (output: B, 2048).
                         If False, exclude avgpool (output: B, 2048, H, W)
        device: Device to load to (default: auto-detect)

    Returns:
        Truncated ResNet50 backbone with contrastive-learned weights

    Example:
        # Load contrastive-pretrained backbone
        backbone = load_contrastive_backbone("logs/.../best_model.pth")

        # Use it like any other truncated model
        model = SiameseNetwork(backbone, embedding_dim=128)
    """
    if not device:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract the full SiameseNetwork state dict
    full_state_dict = checkpoint['model_state_dict']

    # Filter to get only backbone weights (keys starting with 'backbone.')
    backbone_state_dict = {
        k.replace('backbone.0.', ''): v
        for k, v in full_state_dict.items()
        if k.startswith('backbone.0.')
    }

    # Create a sequential model from the state dict
    # We need to reconstruct the backbone architecture
    from torchvision.models import get_model
    temp_model = get_model("resnet50", weights=None, num_classes=1000)

    # Determine how many layers based on single_embedding
    if single_embedding:
        # Include avgpool (layer 9)
        backbone = torch.nn.Sequential(*list(temp_model.children())[:9])
    else:
        # Exclude avgpool (up to layer 8)
        backbone = torch.nn.Sequential(*list(temp_model.children())[:8])

    # Load the contrastive-learned weights
    backbone.load_state_dict(backbone_state_dict)

    # Set to eval mode and move to device
    backbone = backbone.eval()
    backbone = backbone.to(device)

    print(f"✓ Loaded contrastive-pretrained backbone from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return backbone


def load_truncated_model(model_name, device=None, single_embedding=True, truncation_layer=4):
    """
    Load truncated ResNet50 backbone.

    Args:
        model_name: One of 'rgb', 'grey', 'single', 'rad', 'randinit'
        device: Device to load to
        single_embedding: If True, include avgpool (for layer4 only). If False, exclude avgpool.
        truncation_layer: Which ResNet layer to truncate at (3 or 4).
                         - layer3: outputs (B, 1024, 14, 14) - more spatial resolution
                         - layer4: outputs (B, 2048, 7, 7) - default

    Returns:
        Truncated backbone model
    """
    if model_name.lower() =="rgb":
        from torchvision.models import get_model
        # RGB ImageNet
        rgb_weights = torch.load("../models/rgb_3c_model_89.pth", map_location='cpu', weights_only=False)
        rgb_model = get_model("resnet50", weights=None, num_classes=1000)
        rgb_model.load_state_dict(rgb_weights["model"])
        if truncation_layer == 3:
            model = torch.nn.Sequential(*list(rgb_model.children())[:7])  # Up to layer3
        elif single_embedding:
            model = torch.nn.Sequential(*list(rgb_model.children())[:9])  # Layer4 + avgpool
        else:
            model = torch.nn.Sequential(*list(rgb_model.children())[:8])  # Layer4 only
    elif model_name.lower() =="grey":
        from torchvision.models import get_model
        # Greyscale ImageNet
        grey_weights = torch.load("../models/grey_3c_model_89.pth", map_location='cpu', weights_only=False)
        grey_model = get_model("resnet50", weights=None, num_classes=1000)
        grey_model.load_state_dict(grey_weights["model"])
        if truncation_layer == 3:
            model = torch.nn.Sequential(*list(grey_model.children())[:7])  # Up to layer3
        elif single_embedding:
            model = torch.nn.Sequential(*list(grey_model.children())[:9])  # Layer4 + avgpool
        else:
            model = torch.nn.Sequential(*list(grey_model.children())[:8])  # Layer4 only
    elif model_name.lower() =="single":
        from torchvision.models import get_model
        from helpers.models import convert_to_single_channel
        # Single-Channel ImageNet
        # load weights
        single_weights = torch.load("../models/grey_1c_model_89.pth", map_location='cpu', weights_only=False)
        single_model = get_model("resnet50", weights=None, num_classes=1000)
        single_model = convert_to_single_channel(single_model)
        single_model.load_state_dict(single_weights["model"])
        if truncation_layer == 3:
            model = torch.nn.Sequential(*list(single_model.children())[:7])  # Up to layer3
        elif single_embedding:
            model = torch.nn.Sequential(*list(single_model.children())[:9])  # Layer4 + avgpool
        else:
            model = torch.nn.Sequential(*list(single_model.children())[:8])  # Layer4 only
    elif model_name.lower() =="rad":
        # RadImageNet
        from helpers.radimagenet import RadImageNetBackbone
        radimagenet_model = RadImageNetBackbone()
        radimagenet_model.load_state_dict(torch.load("../models/radimagenet_resnet50.pt"))
        if truncation_layer == 3:
            # Patch: RadImageNet wraps layers in Sequential, need to access [0]
            model = torch.nn.Sequential(*list(radimagenet_model.children())[0][:7])  # Up to layer3
        elif single_embedding:
            model = torch.nn.Sequential(*list(radimagenet_model.children())[:9])  # Layer4 + avgpool
        else:
            # Patch: For some reason, this is wrapped in a list or something? This removes the avgpool needed
            model = torch.nn.Sequential(*list(radimagenet_model.children())[0][:8])  # Layer4 only
    elif model_name.lower() == "randinit":
        # no pretraining baseline
        from torchvision.models import get_model
        random_model = get_model("resnet50", weights=None, num_classes=1000)
        if truncation_layer == 3:
            model = torch.nn.Sequential(*list(random_model.children())[:7])  # Up to layer3
        elif single_embedding:
            model = torch.nn.Sequential(*list(random_model.children())[:9])  # Layer4 + avgpool
        else:
            model = torch.nn.Sequential(*list(random_model.children())[:8])  # Layer4 only
    else:
        raise ValueError("Invalid model name! Expects: rgb, grey, single, rad")

    # set to eval mode - safety!!!
    model = model.eval()
    if not device:
        device = get_device()
    # load to device
    model = model.to(device)
    return model
        

class SiameseNetwork(nn.Module):
    """
    Siamese-style network for pair-wise contrastive learning
    Expects truncated to Layer 8 (2048,) pre-trained ResNet50 backbones
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, freeze_backbone=False, num_classes=None):
        super(SiameseNetwork, self).__init__()

        self.backbone = pretrained_backbone

        backbone_dim = 2048

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )

        # Optional classification head (on embeddings, not projections)
        self.classification_head = None
        if num_classes is not None:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        # L2 normalisation layer
        self.l2_norm = nn.functional.normalize

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # def forward_one(self, x):
    #     """Forward pass for one image"""
    #     features = self.backbone(x)  # Output: (batch_size, 2048)
    #     embedding = self.embedding_head(features)  # Output: (batch_size, embedding_dim)
    #     return F.normalize(embedding, p=2, dim=1)  # L2 normalise
    def forward_one(self, x, return_logits=False):
        """
        Forward pass for one image

        Args:
            x: Input image
            return_logits: If True, return classification logits (requires classification_head)
                          If False, return normalized embeddings
        """
        features = self.backbone(x)  # Output: (batch_size, 2048)
        if features.dim() == 4:
            # Ensure global pooling to 1x1 then flatten to (B, 2048)
            if features.shape[-1] != 1 or features.shape[-2] != 1:
                features = F.adaptive_avg_pool2d(features, output_size=1)
            features = torch.flatten(features, 1)
        embedding = self.embedding_head(features)  # Output: (batch_size, embedding_dim)

        if return_logits:
            if self.classification_head is None:
                raise ValueError("Classification head not initialized. Set num_classes during initialization.")
            return self.classification_head(embedding)  # Raw logits (not normalized)
        else:
            return F.normalize(embedding, p=2, dim=1)  # L2 normalize

    def forward(self, x1, x2, return_logits=False):
        """
        Forward pass for image pairs (lung_l, lung_r)

        Args:
            x1, x2: Input image pairs
            return_logits: If True, return classification logits for both images
                          If False, return normalized embeddings
        """
        emb1 = self.forward_one(x1, return_logits=return_logits)
        emb2 = self.forward_one(x2, return_logits=return_logits)
        return emb1, emb2


class SiameseNetworkWithProjection(nn.Module):
    """
    Siamese network with projection head (SimCLR-style).

    Following Chen et al. (2020) - A Simple Framework for Contrastive Learning.
    The projection head improves contrastive learning but is discarded for downstream tasks.

    Architecture:
        Backbone (2048) → Embedding Head (512→128) → Projection Head (128→128→128)

    Usage:
        - During training: Use projection output for contrastive loss
        - For downstream tasks: Use embedding output (before projection)
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, projection_dim=128,
                 freeze_backbone=False, projection_hidden_dim=None, num_classes=None):
        super(SiameseNetworkWithProjection, self).__init__()

        self.backbone = pretrained_backbone
        backbone_dim = 2048

        # Embedding head (representation for downstream tasks)
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )

        # Optional classification head (on embeddings, not projections)
        self.classification_head = None
        if num_classes is not None:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        # Projection head (for contrastive learning only)
        # Following SimCLR: 2-layer MLP with hidden dimension
        if projection_hidden_dim is None:
            projection_hidden_dim = projection_dim

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_one(self, x, return_embedding=False, return_logits=False):
        """
        Forward pass for one image.

        Args:
            x: Input image
            return_embedding: If True, return embedding (before projection).
                            If False, return projection (for contrastive loss).
            return_logits: If True, return classification logits (requires classification_head)

        Returns:
            If return_logits=True: Classification logits (raw, not normalized)
            If return_embedding=True: L2-normalized embedding (for downstream tasks)
            If return_embedding=False: L2-normalized projection (for contrastive loss)
        """
        # Backbone
        features = self.backbone(x)
        if features.dim() == 4:
            if features.shape[-1] != 1 or features.shape[-2] != 1:
                features = F.adaptive_avg_pool2d(features, output_size=1)
            features = torch.flatten(features, 1)

        # Embedding (keep for downstream tasks)
        embedding = self.embedding_head(features)

        if return_logits:
            # Return classification logits (on embeddings, not projections)
            if self.classification_head is None:
                raise ValueError("Classification head not initialized. Set num_classes during initialization.")
            return self.classification_head(embedding)
        elif return_embedding:
            # Return embedding for downstream tasks (linear probe, etc.)
            return F.normalize(embedding, p=2, dim=1)
        else:
            # Return projection for contrastive loss
            projection = self.projection_head(embedding)
            return F.normalize(projection, p=2, dim=1)

    def forward(self, x1, x2, return_embedding=False, return_logits=False):
        """
        Forward pass for image pairs.

        Args:
            x1, x2: Input image pairs
            return_embedding: If True, return embeddings (for downstream).
                            If False, return projections (for contrastive loss).
            return_logits: If True, return classification logits

        Returns:
            (emb1, emb2) or (proj1, proj2) or (logits1, logits2) depending on flags
        """
        emb1 = self.forward_one(x1, return_embedding=return_embedding, return_logits=return_logits)
        emb2 = self.forward_one(x2, return_embedding=return_embedding, return_logits=return_logits)
        return emb1, emb2


class SiameseNetworkSpatial(nn.Module):
    """
    Siamese network that preserves spatial structure through 1x1 convolutions.

    Instead of immediately pooling backbone features to (B, 2048), this network:
    1. Keeps spatial structure: (B, 2048, 7, 7)
    2. Uses 1x1 convolutions to process features spatially
    3. Pools at the very end to get (B, 128)

    This allows spatial correspondence between left/right lung embeddings.

    Architecture:
        Backbone → (B, 2048, 7, 7) → Conv1x1(512) → BN → LeakyReLU →
        Conv1x1(128) → (B, 128, 7, 7) → Global Avg Pool → (B, 128)
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, freeze_backbone=False, num_classes=None):
        super(SiameseNetworkSpatial, self).__init__()

        self.backbone = pretrained_backbone
        backbone_dim = 2048

        # Spatial embedding head using 1x1 convolutions
        self.embedding_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, embedding_dim, kernel_size=1)
        )

        # Optional classification head (on embeddings, not projections)
        self.classification_head = None
        if num_classes is not None:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_one(self, x, return_logits=False):
        """
        Forward pass for one image, preserving spatial structure.

        Args:
            x: Input image (B, C, H, W)
            return_logits: If True, return classification logits (requires classification_head)

        Returns:
            If return_logits=True: Classification logits
            Otherwise: L2-normalized embedding (B, embedding_dim)
        """
        # Backbone - should output (B, 2048, 7, 7)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use single_embedding=False when loading backbone.")

        # Process spatially - (B, 2048, 7, 7) → (B, 128, 7, 7)
        spatial_embedding = self.embedding_head(features)

        # Global average pooling - (B, 128, 7, 7) → (B, 128)
        pooled = F.adaptive_avg_pool2d(spatial_embedding, output_size=1)
        embedding = torch.flatten(pooled, 1)

        if return_logits:
            if self.classification_head is None:
                raise ValueError("Classification head not initialized. Set num_classes during initialization.")
            return self.classification_head(embedding)
        else:
            # L2 normalize
            return F.normalize(embedding, p=2, dim=1)

    def forward_one_spatial(self, x):
        """
        Forward pass returning spatial features before pooling.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            spatial_embedding: Spatial features (B, 128, 7, 7)
        """
        # Backbone - should output (B, 2048, 7, 7)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use single_embedding=False when loading backbone.")

        # Process spatially - (B, 2048, 7, 7) → (B, 128, 7, 7)
        spatial_embedding = self.embedding_head(features)

        return spatial_embedding

    def forward(self, x1, x2, return_logits=False):
        """Forward pass for image pairs"""
        emb1 = self.forward_one(x1, return_logits=return_logits)
        emb2 = self.forward_one(x2, return_logits=return_logits)
        return emb1, emb2


class SiameseNetworkWithProjectionSpatial(nn.Module):
    """
    Siamese network with projection head that preserves spatial structure.

    Combines spatial processing (1x1 convs) with SimCLR-style projection head.

    Architecture:
        Backbone → (B, 2048, 7, 7) → Spatial Embedding Head → (B, 128, 7, 7) →
        Pool → (B, 128) → Projection Head → (B, projection_dim)

    Usage:
        - During training: Use projection output for contrastive loss
        - For downstream tasks: Use embedding output (before projection)
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, projection_dim=128,
                 freeze_backbone=False, projection_hidden_dim=None, num_classes=None):
        super(SiameseNetworkWithProjectionSpatial, self).__init__()

        self.backbone = pretrained_backbone
        backbone_dim = 2048

        # Spatial embedding head using 1x1 convolutions
        self.embedding_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, embedding_dim, kernel_size=1)
        )

        # Optional classification head (on embeddings, not projections)
        self.classification_head = None
        if num_classes is not None:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        # Projection head (for contrastive learning only)
        # Following SimCLR: 2-layer MLP with hidden dimension
        if projection_hidden_dim is None:
            projection_hidden_dim = projection_dim

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_one(self, x, return_embedding=False, return_logits=False):
        """
        Forward pass for one image.

        Args:
            x: Input image
            return_embedding: If True, return embedding (before projection).
                            If False, return projection (for contrastive loss).
            return_logits: If True, return classification logits (requires classification_head)

        Returns:
            If return_logits=True: Classification logits
            If return_embedding=True: L2-normalized embedding (for downstream tasks)
            If return_embedding=False: L2-normalized projection (for contrastive loss)
        """
        # Backbone - should output (B, 2048, 7, 7)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use single_embedding=False when loading backbone.")

        # Process spatially - (B, 2048, 7, 7) → (B, 128, 7, 7)
        spatial_embedding = self.embedding_head(features)

        # Global average pooling - (B, 128, 7, 7) → (B, 128)
        pooled = F.adaptive_avg_pool2d(spatial_embedding, output_size=1)
        embedding = torch.flatten(pooled, 1)

        if return_logits:
            if self.classification_head is None:
                raise ValueError("Classification head not initialized. Set num_classes during initialization.")
            return self.classification_head(embedding)
        elif return_embedding:
            # Return embedding for downstream tasks
            return F.normalize(embedding, p=2, dim=1)
        else:
            # Return projection for contrastive loss
            projection = self.projection_head(embedding)
            return F.normalize(projection, p=2, dim=1)

    def forward_one_spatial(self, x):
        """
        Forward pass returning spatial features before pooling.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            spatial_embedding: Spatial features (B, 128, 7, 7)
        """
        # Backbone - should output (B, 2048, 7, 7)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use single_embedding=False when loading backbone.")

        # Process spatially - (B, 2048, 7, 7) → (B, 128, 7, 7)
        spatial_embedding = self.embedding_head(features)

        return spatial_embedding

    def forward(self, x1, x2, return_embedding=False, return_logits=False):
        """
        Forward pass for image pairs.

        Args:
            x1, x2: Input image pairs
            return_embedding: If True, return embeddings (for downstream).
                            If False, return projections (for contrastive loss).
            return_logits: If True, return classification logits

        Returns:
            (emb1, emb2) or (proj1, proj2) or (logits1, logits2) depending on flags
        """
        emb1 = self.forward_one(x1, return_embedding=return_embedding, return_logits=return_logits)
        emb2 = self.forward_one(x2, return_embedding=return_embedding, return_logits=return_logits)
        return emb1, emb2


class SiameseNetworkSpatialEarly(nn.Module):
    """
    Siamese network that preserves spatial structure using early truncation (layer3).

    Truncates at ResNet layer3 instead of layer4, giving:
    - More spatial resolution: 14×14 instead of 7×7
    - Fewer channels: 1024 instead of 2048
    - Earlier features with more spatial detail

    Architecture:
        Backbone → (B, 1024, 14, 14) → Conv1x1(512) → BN → LeakyReLU →
        Conv1x1(128) → (B, 128, 14, 14) → Global Avg Pool → (B, 128)
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, freeze_backbone=False, num_classes=None):
        super(SiameseNetworkSpatialEarly, self).__init__()

        self.backbone = pretrained_backbone
        backbone_dim = 1024  # layer3 output

        # Spatial embedding head using 1x1 convolutions
        self.embedding_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, embedding_dim, kernel_size=1)
        )

        # Optional classification head (on embeddings, not projections)
        self.classification_head = None
        if num_classes is not None:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_one(self, x, return_logits=False):
        """
        Forward pass for one image, preserving spatial structure.

        Args:
            x: Input image (B, C, H, W)
            return_logits: If True, return classification logits (requires classification_head)

        Returns:
            If return_logits=True: Classification logits
            Otherwise: L2-normalized embedding (B, embedding_dim)
        """
        # Backbone - should output (B, 1024, 14, 14)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use early truncation when loading backbone.")

        # Verify correct dimensions
        if features.shape[1] != 1024:
            raise ValueError(f"Expected 1024 channels from layer3, got {features.shape[1]}. Use backbone truncated at layer3.")

        # Process spatially - (B, 1024, 14, 14) → (B, 128, 14, 14)
        spatial_embedding = self.embedding_head(features)

        # Global average pooling - (B, 128, 14, 14) → (B, 128)
        pooled = F.adaptive_avg_pool2d(spatial_embedding, output_size=1)
        embedding = torch.flatten(pooled, 1)

        if return_logits:
            if self.classification_head is None:
                raise ValueError("Classification head not initialized. Set num_classes during initialization.")
            return self.classification_head(embedding)
        else:
            # L2 normalize
            return F.normalize(embedding, p=2, dim=1)

    def forward_one_spatial(self, x):
        """
        Forward pass returning spatial features before pooling.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            spatial_embedding: Spatial features (B, 128, 14, 14)
        """
        # Backbone - should output (B, 1024, 14, 14)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use early truncation when loading backbone.")

        # Verify correct dimensions
        if features.shape[1] != 1024:
            raise ValueError(f"Expected 1024 channels from layer3, got {features.shape[1]}. Use backbone truncated at layer3.")

        # Process spatially - (B, 1024, 14, 14) → (B, 128, 14, 14)
        spatial_embedding = self.embedding_head(features)

        return spatial_embedding

    def forward(self, x1, x2, return_logits=False):
        """Forward pass for image pairs"""
        emb1 = self.forward_one(x1, return_logits=return_logits)
        emb2 = self.forward_one(x2, return_logits=return_logits)
        return emb1, emb2


class SiameseNetworkWithProjectionSpatialEarly(nn.Module):
    """
    Siamese network with projection head using early truncation (layer3).

    Combines early spatial processing (14×14 resolution) with SimCLR-style projection head.

    Architecture:
        Backbone → (B, 1024, 14, 14) → Spatial Embedding Head → (B, 128, 14, 14) →
        Pool → (B, 128) → Projection Head → (B, projection_dim)

    Usage:
        - During training: Use projection output for contrastive loss
        - For downstream tasks: Use embedding output (before projection)
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, projection_dim=128,
                 freeze_backbone=False, projection_hidden_dim=None, num_classes=None):
        super(SiameseNetworkWithProjectionSpatialEarly, self).__init__()

        self.backbone = pretrained_backbone
        backbone_dim = 1024  # layer3 output

        # Spatial embedding head using 1x1 convolutions
        self.embedding_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, embedding_dim, kernel_size=1)
        )

        # Optional classification head (on embeddings, not projections)
        self.classification_head = None
        if num_classes is not None:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        # Projection head (for contrastive learning only)
        # Following SimCLR: 2-layer MLP with hidden dimension
        if projection_hidden_dim is None:
            projection_hidden_dim = projection_dim

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_one(self, x, return_embedding=False, return_logits=False):
        """
        Forward pass for one image.

        Args:
            x: Input image
            return_embedding: If True, return embedding (before projection).
                            If False, return projection (for contrastive loss).
            return_logits: If True, return classification logits (requires classification_head)

        Returns:
            If return_logits=True: Classification logits
            If return_embedding=True: L2-normalized embedding (for downstream tasks)
            If return_embedding=False: L2-normalized projection (for contrastive loss)
        """
        # Backbone - should output (B, 1024, 14, 14)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use early truncation when loading backbone.")

        # Verify correct dimensions
        if features.shape[1] != 1024:
            raise ValueError(f"Expected 1024 channels from layer3, got {features.shape[1]}. Use backbone truncated at layer3.")

        # Process spatially - (B, 1024, 14, 14) → (B, 128, 14, 14)
        spatial_embedding = self.embedding_head(features)

        # Global average pooling - (B, 128, 14, 14) → (B, 128)
        pooled = F.adaptive_avg_pool2d(spatial_embedding, output_size=1)
        embedding = torch.flatten(pooled, 1)

        if return_logits:
            if self.classification_head is None:
                raise ValueError("Classification head not initialized. Set num_classes during initialization.")
            return self.classification_head(embedding)
        elif return_embedding:
            # Return embedding for downstream tasks
            return F.normalize(embedding, p=2, dim=1)
        else:
            # Return projection for contrastive loss
            projection = self.projection_head(embedding)
            return F.normalize(projection, p=2, dim=1)

    def forward_one_spatial(self, x):
        """
        Forward pass returning spatial features before pooling.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            spatial_embedding: Spatial features (B, 128, 14, 14)
        """
        # Backbone - should output (B, 1024, 14, 14)
        features = self.backbone(x)

        # Handle case where backbone already pooled
        if features.dim() == 2:
            raise ValueError("Backbone output is already pooled! Use early truncation when loading backbone.")

        # Verify correct dimensions
        if features.shape[1] != 1024:
            raise ValueError(f"Expected 1024 channels from layer3, got {features.shape[1]}. Use backbone truncated at layer3.")

        # Process spatially - (B, 1024, 14, 14) → (B, 128, 14, 14)
        spatial_embedding = self.embedding_head(features)

        return spatial_embedding

    def forward(self, x1, x2, return_embedding=False, return_logits=False):
        """
        Forward pass for image pairs.

        Args:
            x1, x2: Input image pairs
            return_embedding: If True, return embeddings (for downstream).
                            If False, return projections (for contrastive loss).
            return_logits: If True, return classification logits

        Returns:
            (emb1, emb2) or (proj1, proj2) or (logits1, logits2) depending on flags
        """
        emb1 = self.forward_one(x1, return_embedding=return_embedding, return_logits=return_logits)
        emb2 = self.forward_one(x2, return_embedding=return_embedding, return_logits=return_logits)
        return emb1, emb2


class ContrastiveNetwork(nn.Module):
    """
    Network for supervised contrastive learning (Khosla et al.)
    Adapted for lung image dataset with pre-trained backbones
    """
    def __init__(self, pretrained_backbone, embedding_dim=128, projection_dim=128, freeze_backbone=False):
        super(ContrastiveNetwork, self).__init__()
        self.backbone = pretrained_backbone
        
        # Your backbones output 2048-dim features after AdaptiveAvgPool2d
        backbone_dim = 2048
        
        # Feature extraction head  
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Projection head (common in contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        """Forward pass for batch of images"""
        features = self.backbone(x)
        features = self.feature_head(features)
        projections = self.projector(features)
        # L2 normalize for contrastive learning
        return F.normalize(projections, dim=1)