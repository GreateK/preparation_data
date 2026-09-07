import re
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
from pathlib import Path


class TerrainDataset(Dataset):
    def __init__(self, 
                images_dir='./training_data/images', 
                masks_dir='./training_data/masks/csv',
                transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.pairs = self._build_indexed_pairs()
        print(f"Successfully verified and paired {len(self.pairs)} items based on index.")
        
    def _extract_index(self, filename: Path) -> str:
            match = re.search(r'\d+', filename.name)
            return match.group() if match else None

    def _build_indexed_pairs(self) -> list:

        mask_map = {}
        for mask_path in self.masks_dir.iterdir():
            if mask_path.is_file() and mask_path.suffix == '.csv':
                idx = self._extract_index(mask_path)
                if idx is not None:
                    mask_map[idx] = mask_path

        pairs = []
        for img_path in self.images_dir.iterdir():
            if img_path.is_file() and img_path.suffix == '.png':
                img_idx = self._extract_index(img_path)
                
                if img_idx in mask_map:
                    pairs.append((img_path, mask_map[img_idx]))
                else:
                    print(f"Warning: No matching mask found for image index {img_idx} ({img_path.name})")
                    
        if not pairs:
            raise RuntimeError("No matching image-mask index pairs found!")
            
        return pairs


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask_np = np.loadtxt(mask_path, delimiter=',', dtype=np.int64)
        
        mask_tensor = torch.from_numpy(mask_np) 
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback if no transform is passed: convert PIL image to tensor manually
            image = transforms.functional.to_image(image)
            image = transforms.functional.to_dtype(image, torch.float32, scale=True)
            
        return image, mask_tensor

    
total_frames = TerrainDataset()

dataset_len = len(total_frames)
print(f"Dataset length returned: {dataset_len}")

# Define how to process the PNG image
img_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True) # Converts to float and scales pixels to [0, 1]
])

# Instantiate dataset
dataset = TerrainDataset(
    images_dir='./training_data/images',
    masks_dir='./training_data/masks/csv',
    transform=img_transform
)

image_tensor, mask_tensor = dataset[2]

print("--- Tensor Sanity Check ---")
print(f"Image - Dtype: {image_tensor.dtype}, Shape: {image_tensor.shape}")
print(f"Mask  - Dtype: {mask_tensor.dtype}, Shape: {mask_tensor.shape}")
print(f"Unique Labels in this Mask: {torch.unique(mask_tensor).tolist()}")

train_loader = DataLoader(
    dataset, 
    batch_size=4,       # Adjust based on your GPU VRAM (1024x1024 is quite large!)
    shuffle=True,       # Shuffle data for training
    num_workers=0,      # Multi-process data loading
    drop_last=True      # Drop incomplete batches
)

# Test the batching shape
for batch_images, batch_masks in train_loader:
    print("\n--- Batch Shapes ---")
    print("Batch Images Shape:", batch_images.shape)  # Expected: [4, 3, 1024, 1024]
    print("Batch Masks Shape:", batch_masks.shape)    # Expected: [4, 1024, 1024]
    break

# Learning_Rate=1e-5
# width=height=1024
# batch_size = 4

