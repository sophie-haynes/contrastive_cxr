from torchvision.transforms import v2
from torch import float32 as tfloat32
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import torch
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


class ImagePairDataset(Dataset):
    """Custom dataset for loading image pairs organized as:
    root/class_name/pair_name/lung_l.png
    root/class_name/pair_name/lung_r.png
    """

    def __init__(self, root, transform=None, symmetrical_transforms=False,
                 image_names=('lung_l.png', 'lung_r.png'), class_to_idx=None,
                 cache_in_ram=False):
        """
        Args:
            root (str): Root directory path
            transform: torchvision transforms to apply to both images
            symmetrical_transforms (bool): apply identical transforms to pairs
            image_names (tuple): Names of the two images in each pair folder
            class_to_idx (dict): Optional mapping of class names to indices.
                                 If None, will use alphabetical ordering.
            cache_in_ram (bool): If True, preload base images into RAM as PIL.
        """
        self.root = root
        self.transform = transform
        self.symmetrical_transforms = symmetrical_transforms
        self.image_names = image_names
        self.predefined_class_to_idx = class_to_idx

        # caching
        self.cache_in_ram = cache_in_ram
        self._image_cache = {} if cache_in_ram else None

        # Build the dataset index
        self.pairs = []
        self.class_to_idx = {}
        self._build_dataset()

        # Optionally cache all images now
        if self.cache_in_ram:
            self._preload_images()

    def _build_dataset(self):
        """Build list of all image pairs and create class mappings"""
        classes = sorted([d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d))])

        # Use predefined mapping if provided, otherwise use alphabetical
        if self.predefined_class_to_idx:
            self.class_to_idx = self.predefined_class_to_idx.copy()
            # Verify all classes in dataset are in the predefined mapping
            missing_classes = set(classes) - set(self.class_to_idx.keys())
            if missing_classes:
                raise ValueError(f"Classes found in dataset but not in predefined mapping: {missing_classes}")
            # Warn about classes in mapping but not in dataset
            extra_classes = set(self.class_to_idx.keys()) - set(classes)
            if extra_classes:
                print(f"Warning: Classes in predefined mapping but not found in dataset: {extra_classes}")
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for class_name in tqdm(classes, desc="Loading dataset"):
            class_path = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]

            # Get all pair directories in this class
            pair_dirs = [d for d in os.listdir(class_path)
                         if os.path.isdir(os.path.join(class_path, d))]

            for pair_name in pair_dirs:
                pair_path = os.path.join(class_path, pair_name)

                # Check if both images exist
                lungl_path = os.path.join(pair_path, self.image_names[0])
                lungr_path = os.path.join(pair_path, self.image_names[1])

                if os.path.exists(lungl_path) and os.path.exists(lungr_path):
                    self.pairs.append({
                        'class_name': class_name,
                        'class_idx': class_idx,
                        'pair_name': pair_name,
                        'lungl_path': lungl_path,
                        'lungr_path': lungr_path
                    })
                else:
                    print(f"Warning: Missing images in {pair_path}")

    def _preload_images(self):
        """Load all base images into RAM as PIL RGB."""
        print("Caching base images into RAM...")
        unique_paths = set()
        for p in self.pairs:
            unique_paths.add(p['lungl_path'])
            unique_paths.add(p['lungr_path'])

        skipped = 0
        for path in tqdm(sorted(unique_paths), desc="Caching"):
            try:
                with Image.open(path) as im:
                    self._image_cache[path] = im.convert('RGB').copy()
            except Exception as e:
                skipped += 1
                print(f"Error caching {path}: {e}")
        print(f"Cached {len(self._image_cache)} images"
              + (f" (skipped {skipped})" if skipped else ""))

    def _load_image(self, path):
        """Return a PIL RGB image, from cache if enabled."""
        if self.cache_in_ram and path in self._image_cache:
            # copy to avoid in-place mutation by transforms
            return self._image_cache[path].copy()
        # fallback: disk load
        with Image.open(path) as im:
            return im.convert('RGB')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (lungl, lungr, class_idx, lungl_path, lungr_path)
        """
        pair_info = self.pairs[idx]

        # Load images (from cache if enabled)
        lungl = self._load_image(pair_info['lungl_path'])
        lungr = self._load_image(pair_info['lungr_path'])

        # Apply transforms (unchanged logic)
        if self.transform:
            if self.symmetrical_transforms:
                # Generate a random seed for this pair
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
            lungl = self.transform(lungl)
            # reproduce same random transforms by setting seeds
            if self.symmetrical_transforms:
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
            lungr = self.transform(lungr)

        return (
            lungl,
            lungr,
            pair_info['class_idx'],
            pair_info['lungl_path'],
            pair_info['lungr_path']
        )

    def get_class_name(self, class_idx):
        """Get class name from class index"""
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        return idx_to_class[class_idx]

    def get_pairs_by_class(self, class_name):
        """Get all pairs for a specific class"""
        return [pair for pair in self.pairs if pair['class_name'] == class_name]

    def print_dataset_info(self):
        """Print dataset statistics"""
        print(f"Total pairs: {len(self.pairs)}")
        print(f"Number of classes: {len(self.class_to_idx)}")
        print(f"Class mapping: {self.class_to_idx}")

        # Count pairs per class
        class_counts = {}
        for pair in self.pairs:
            class_name = pair['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nPairs per class:")
        for class_name, count in sorted(class_counts.items()):
            class_idx = self.class_to_idx[class_name]
            print(f"  {class_name} (idx={class_idx}): {count}")


class SingleImageDataset(Dataset):
    """Custom dataset for loading single full CXR images organized as:
    root/class_name/cxr_name.png
    """

    def __init__(self, root, transform=None, class_to_idx=None,
                 cache_in_ram=False):
        """
        Args:
            root (str): Root directory path
            transform: torchvision transforms to apply to the image
            class_to_idx (dict): Optional mapping of class names to indices.
                                 If None, will use alphabetical ordering.
            cache_in_ram (bool): If True, preload base images into RAM as PIL.
        """
        self.root = root
        self.transform = transform
        self.predefined_class_to_idx = class_to_idx

        # caching
        self.cache_in_ram = cache_in_ram
        self._image_cache = {} if cache_in_ram else None

        # Build the dataset index
        self.samples = []
        self.class_to_idx = {}
        self._build_dataset()

        # Optionally cache all images now
        if self.cache_in_ram:
            self._preload_images()

    def _build_dataset(self):
        """Build list of all images and create class mappings"""
        classes = sorted([d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d))])

        # Use predefined mapping if provided, otherwise use alphabetical
        if self.predefined_class_to_idx:
            self.class_to_idx = self.predefined_class_to_idx.copy()
            # Verify all classes in dataset are in the predefined mapping
            missing_classes = set(classes) - set(self.class_to_idx.keys())
            if missing_classes:
                raise ValueError(f"Classes found in dataset but not in predefined mapping: {missing_classes}")
            # Warn about classes in mapping but not in dataset
            extra_classes = set(self.class_to_idx.keys()) - set(classes)
            if extra_classes:
                print(f"Warning: Classes in predefined mapping but not found in dataset: {extra_classes}")
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for class_name in tqdm(classes, desc="Loading dataset"):
            class_path = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect image files in this class
            try:
                filenames = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
            except FileNotFoundError:
                filenames = []

            for fname in filenames:
                img_path = os.path.join(class_path, fname)
                self.samples.append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'image_path': img_path
                })

    def _preload_images(self):
        """Load all base images into RAM as PIL RGB."""
        print("Caching base images into RAM...")
        unique_paths = {s['image_path'] for s in self.samples}

        skipped = 0
        for path in tqdm(sorted(unique_paths), desc="Caching"):
            try:
                with Image.open(path) as im:
                    self._image_cache[path] = im.convert('RGB').copy()
            except Exception as e:
                skipped += 1
                print(f"Error caching {path}: {e}")
        print(f"Cached {len(self._image_cache)} images"
              + (f" (skipped {skipped})" if skipped else ""))

    def _load_image(self, path):
        """Return a PIL RGB image, from cache if enabled."""
        if self.cache_in_ram and path in self._image_cache:
            # copy to avoid in-place mutation by transforms
            return self._image_cache[path].copy()
        # fallback: disk load
        with Image.open(path) as im:
            return im.convert('RGB')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image_tensor, class_idx, image_path)
        """
        info = self.samples[idx]

        # Load image (from cache if enabled)
        img = self._load_image(info['image_path'])

        # Apply transforms
        if self.transform:
            # keep RNG usage consistent with your style (seed control not needed for single images)
            img = self.transform(img)

        return (
            img,
            info['class_idx'],
            info['image_path']
        )

    def get_class_name(self, class_idx):
        """Get class name from class index"""
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        return idx_to_class[class_idx]

    def get_images_by_class(self, class_name):
        """Get all image records for a specific class"""
        return [s for s in self.samples if s['class_name'] == class_name]

    def print_dataset_info(self):
        """Print dataset statistics"""
        print(f"Total images: {len(self.samples)}")
        print(f"Number of classes: {len(self.class_to_idx)}")
        print(f"Class mapping: {self.class_to_idx}")

        # Count images per class
        class_counts = {}
        for s in self.samples:
            class_name = s['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nImages per class:")
        for class_name, count in sorted(class_counts.items()):
            class_idx = self.class_to_idx[class_name]
            print(f"  {class_name} (idx={class_idx}): {count}")


def load_image_pair_dataset(dataset_path, crop_size=512, batch_size=4,
                            shuffle=True, transform=None, image_names=('lung_l.png', 'lung_r.png'),
                            symmetrical_transforms=False, single=False, class_to_idx=None,
                            num_workers=2, cache_in_ram=False):
    """
    Wrapper function to load image pair dataset with DataLoader

    Args:
        dataset_path (str): Path to dataset root
        crop_size (int): Size for image cropping/resizing
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        transform: Custom transform, if None will use default ResNet50 transforms
        image_names (tuple): Names of the two images in each pair folder
        symmetrical_transforms (bool): Apply identical transforms to pair
        single (bool): Output single channel image
        class_to_idx (dict): Optional mapping of class names to indices
        num_workers (int): Number of workers for dataloader, if caching, set to 1
        cache_in_ram (bool): Pre-load and cache dataset in RAM to speed up image loading

    Returns:
        DataLoader: Configured DataLoader for the dataset
    """

    # Default ResNet50 transforms if none provided
    if transform is None:

        channels = 1 if single else 3
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Grayscale(channels),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])

    dataset = ImagePairDataset(
        root=dataset_path,
        transform=transform,
        symmetrical_transforms=symmetrical_transforms,
        image_names=image_names,
        class_to_idx=class_to_idx,
        cache_in_ram=cache_in_ram

    )

    # Print dataset info
    dataset.print_dataset_info()

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_single_image_dataset(dataset_path, crop_size=512, batch_size=4,
                              shuffle=True, transform=None, single=False,
                              class_to_idx=None, num_workers=2, cache_in_ram=False):
    """
    Wrapper function to load single-image dataset with DataLoader

    Args:
        dataset_path (str): Path to dataset root
        crop_size (int): Size for image resizing
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        transform: Custom transform, if None will use default ResNet-like transforms
        single (bool): Output single-channel grayscale (1) vs 3-channel grayscale (3)
        class_to_idx (dict): Optional mapping of class names to indices
        num_workers (int): Number of workers for dataloader (if caching, consider 1)
        cache_in_ram (bool): Pre-load and cache dataset in RAM to speed up image loading

    Returns:
        DataLoader: Configured DataLoader for the dataset
    """
    # Default transforms if none provided
    if transform is None:
        channels = 1 if single else 3
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Grayscale(channels),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])

    dataset = SingleImageDataset(
        root=dataset_path,
        transform=transform,
        class_to_idx=class_to_idx,
        cache_in_ram=cache_in_ram
    )

    # Print dataset info
    dataset.print_dataset_info()

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
