import torch 
from torch.utils.data import Dataset
from pathlib import Path

class TerrainDataset(Dataset):
    def __init__(self, data_dir='./training_data/masks'):
        self.dir_path = Path(data_dir)
        self.file_count = sum(1 for item in self.dir_path.iterdir() if item.is_file())
        print(f"total files loaded: {self.file_count}")

    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        pass

total_frames = TerrainDataset()

dataset_len = len(total_frames)
print(f"Dataset length returned: {dataset_len}")

