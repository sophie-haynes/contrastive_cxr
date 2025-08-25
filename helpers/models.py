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

def load_truncated_model(model_name, device=None):
    if model_name.lower() =="rgb":
        from torchvision.models import get_model
        # RGB ImageNet
        rgb_weights = torch.load("../models/rgb_3c_model_89.pth", map_location='cpu', weights_only=False)
        rgb_model = get_model("resnet50", weights=None, num_classes=1000)
        rgb_model.load_state_dict(rgb_weights["model"])
        model = torch.nn.Sequential(*list(rgb_model.children())[:9])
    elif model_name.lower() =="grey":
        from torchvision.models import get_model
        # Greyscale ImageNet
        grey_weights = torch.load("../models/grey_3c_model_89.pth", map_location='cpu', weights_only=False)
        grey_model = get_model("resnet50", weights=None, num_classes=1000)
        grey_model.load_state_dict(grey_weights["model"])
        model = torch.nn.Sequential(*list(grey_model.children())[:9])
    elif model_name.lower() =="single":
        from torchvision.models import get_model
        # Single-Channel ImageNet
        # load weights
        single_weights = torch.load("../models/grey_1c_model_89.pth", map_location='cpu', weights_only=False)
        single_model = get_model("resnet50", weights=None, num_classes=1000)
        single_model = helpers.models.convert_to_single_channel(single_model)
        single_model.load_state_dict(single_weights["model"])
        model = torch.nn.Sequential(*list(single_model.children())[:9])
    elif model_name.lower() =="rad":
        # RadImageNet
        from helpers.radimagenet import RadImageNetBackbone
        radimagenet_model = RadImageNetBackbone()
        radimagenet_model.load_state_dict(torch.load("../models/radimagenet_resnet50.pt"))
        model = torch.nn.Sequential(*list(radimagenet_model.children())[:9])
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
    def __init__(self, pretrained_backbone, embedding_dim=128, freeze_backbone=False):
        super(SiameseNetwork, self).__init__()
        
        self.backbone = pretrained_backbone
        
        backbone_dim = 2048
        
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        
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
    def forward_one(self, x):
        """Forward pass for one image"""
        features = self.backbone(x)  # Output: (batch_size, 2048)
        if features.dim() == 4:
            # Ensure global pooling to 1x1 then flatten to (B, 2048)
            if features.shape[-1] != 1 or features.shape[-2] != 1:
                features = F.adaptive_avg_pool2d(features, output_size=1)
            features = torch.flatten(features, 1)
        embedding = self.embedding_head(features)  # Output: (batch_size, embedding_dim)
        return F.normalize(embedding, p=2, dim=1)  # L2 normalize
    
    def forward(self, x1, x2):
        """Forward pass for image pairs (lung_l, lung_r)"""
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2