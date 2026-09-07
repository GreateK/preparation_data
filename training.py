import torch 
from torch.utils.data import Dataset
from pathlib import Path

class TerrainDataset(Dataset):
    def __init__(self, images_dir='./training_data/masks', masks_dir='./training_data/masks/csv'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        self.image_files = sorted([f for f in self.images_dir.iterdir() if f.is_file() and f.suffix == '.png'])
        self.mask_files = sorted([f for f in self.masks_dir.iterdir() if f.is_file() and f.suffix == '.csv'])

        # make sure there is an eq num of img and csv
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch! Found {len(self.image_files)} images but {len(self.mask_files)} masks."
            
        self.pairs = list(zip(self.image_files, self.mask_files))
        
        print(f"Successfully paired {len(self.pairs)} image-mask sets.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        return str(img_path), str(mask_path)
total_frames = TerrainDataset()

dataset_len = len(total_frames)
print(f"Dataset length returned: {dataset_len}")

