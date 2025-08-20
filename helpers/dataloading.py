from torchvision.transforms import v2
from torch import float32 as tfloat32

# crop dictionary of calculated dataset means and std devs
CROP_DICT = {
    # data      mean         std
    'cxr14': [[162.7414], [44.0700]],
    'openi': [[157.6150], [41.8371]],
    'jsrt': [[161.7889], [41.3950]],
    'padchest': [[160.3638], [44.8449]],
}

# arch segmented dictionary of calculated dataset means and std devs
ARCH_SEG_DICT = {
    # data       mean        std
    'cxr14': [[128.2716], [76.7148]],
    'openi': [[127.7211], [69.7704]],
    'jsrt': [[139.9666], [72.4017]],
    'padchest': [[129.5006], [72.6308]],
    'padcxr14': [[128.8861], [74.6728]]
}

# lung segmented dictionary of calculated dataset means and std devs
LUNG_SEG_DICT = {
    # data       mean        std
    'cxr14': [[60.6809], [68.9660]],
    'openi': [[60.5483], [66.5276]],
    'jsrt': [[66.5978], [72.6493]],
    'padchest': [[60.5482], [66.5276]],
    'padcxr14': [[60.61455], [67.7468]]
}


def get_cxr_eval_transforms(crop_size, normalise):
    """
    Returns evaluation transforms for CXR images. Pass in target 
    crop size and the normalisation method for target dataset.
    """
    cxr_transform_list = [
        v2.ToImage(),
        v2.Resize(size=crop_size, antialias=True),
        v2.ToDtype(tfloat32, scale=False),
        normalise
    ]
    return v2.Compose(cxr_transform_list)


def get_cxr_single_eval_transforms(crop_size, normalise):
    """
    Returns evaluation transforms for single channel output CXR 
    images. Pass in target crop size and the normalisation method 
    for target dataset.
    """
    cxr_transform_list = [
        v2.ToImage(),
        v2.Grayscale(1),
        v2.Resize(size=crop_size, antialias=True),
        v2.ToDtype(tfloat32, scale=False),
        normalise,
    ]
    return v2.Compose(cxr_transform_list)


def get_cxr_dataset_normalisation(dataset, process):
    """
    Returns normalisation transform for given dataset/config. Pass 
    in dataset name and the image processing method used.

    Args:
    - dataset (str): Name of CXR dataset. Expects ("cxr14", "padchest", "openi", "jsrt").
    - process (str): Name of CXR processing applied. Expects ("crop", "arch", "lung").

    Returns:
    - torchvision.transform.V2 normalize method.

    """
    if process.lower() not in ("crop", "arch", "lung"):
        raise ValueError(f"Unexpected CXR processing type: \
            {process}! Please choose from (crop, arch, lung).")
    else:
        if dataset.lower() not in ("cxr14", "padchest", "openi", "jsrt"):
            raise ValueError(f"Unexpected CXR dataset type: \
                {dataset}! Please choose from (cxr14, padchest, \
                openi, jsrt).")
        else:
            return v2.Normalize(CROP_DICT[dataset.lower()][0],
                                CROP_DICT[dataset.lower()][1]) \
                if process.lower() == "crop" \
                else \
                v2.Normalize(ARCH_SEG_DICT[dataset.lower()][0],
                             ARCH_SEG_DICT[dataset.lower()][1]) \
                if process.lower() == "arch" \
                else v2.Normalize(LUNG_SEG_DICT[dataset.lower()][0],
                                  LUNG_SEG_DICT[dataset.lower()][1])

