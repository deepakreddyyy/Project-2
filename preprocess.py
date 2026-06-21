import torch
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

# CIFAR-10 stats (mean and std for normalization)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

class HFCifar10Dataset(Dataset):
    """
    Wrapper for Hugging Face CIFAR-10 dataset to interface with PyTorch DataLoaders.
    """
    def __init__(self, split='train', transform=None):
        print(f"Loading Hugging Face CIFAR-10 dataset ({split} split)...")
        # Load dataset from Hugging Face uoft-cs/cifar10
        self.dataset = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform
        print(f"Loaded {len(self.dataset)} samples.")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['img'] # This is a PIL Image object
        label = item['label'] # Integer label (0-9)
        
        # Apply transformation if specified
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_train_transforms():
    """
    Returns the training transforms including data augmentation.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

def get_val_transforms():
    """
    Returns the validation/test transforms (no data augmentation).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

def preprocess_image_for_inference(image_path_or_pil):
    """
    Loads and preprocesses an image from a path or a PIL image object for model inference.
    Resizes the image to 32x32 pixels, converts to tensor, and normalizes it.
    Returns:
        torch.Tensor of shape (1, 3, 32, 32)
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert('RGB')
    else:
        image = image_path_or_pil.convert('RGB')
        
    # Resize to 32x32 as required by CIFAR-10 trained model
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    # Apply transforms and add batch dimension (1, 3, 32, 32)
    tensor = transform(image).unsqueeze(0)
    return tensor
