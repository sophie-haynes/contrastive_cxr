from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import torch

class ImagePairDataset(Dataset):
    """Custom dataset for loading image pairs organized as:
    root/class_name/pair_name/lung_l.png
    root/class_name/pair_name/lung_r.png
    """
    
    def __init__(self, root, transform=None, image_names=('lung_l.png', 'lung_r.png')):
        """
        Args:
            root (str): Root directory path
            transform: torchvision transforms to apply to both images
            image_names (tuple): Names of the two images in each pair folder
        """
        self.root = root
        self.transform = transform
        self.image_names = image_names
        
        # Build the dataset index
        self.pairs = []
        self.class_to_idx = {}
        self._build_dataset()
        
    def _build_dataset(self):
        """Build list of all image pairs and create class mappings"""
        classes = sorted([d for d in os.listdir(self.root) 
                         if os.path.isdir(os.path.join(self.root, d))])
        
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
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (lungl, lungr, class_idx, lungl_path, lungr_path)
        """
        pair_info = self.pairs[idx]

        print(f"PAIR INFO: {pair_info}")
        # Load images
        lungl = Image.open(pair_info['lungl_path']).convert('RGB')
        lungr = Image.open(pair_info['lungl_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            lungl = self.transform(lungl)
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
        
        # Count pairs per class
        class_counts = {}
        for pair in self.pairs:
            class_name = pair['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nPairs per class:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")


def load_image_pair_dataset(dataset_path, crop_size=512, batch_size=4, 
                           shuffle=True, transform=None, image_names=('lung_l.png', 'lung_r.png')):
    """
    Wrapper function to load image pair dataset with DataLoader
    
    Args:
        dataset_path (str): Path to dataset root
        crop_size (int): Size for image cropping/resizing
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        transform: Custom transform, if None will use default ResNet50 transforms
        image_names (tuple): Names of the two images in each pair folder
    
    Returns:
        DataLoader: Configured DataLoader for the dataset
    """
    
    # Default ResNet50 transforms if none provided
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],  # ImageNet means
            #     std=[0.229, 0.224, 0.225]    # ImageNet stds
            # )
        ])
    
    dataset = ImagePairDataset(
        root=dataset_path,
        transform=transform,
        image_names=image_names
    )
    
    # Print dataset info
    dataset.print_dataset_info()
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
