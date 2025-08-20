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










