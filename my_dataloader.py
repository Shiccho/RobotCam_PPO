from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
import random

class OrderedDataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        super().__init__()
        self.transform = transform
        self.data = Path(root).glob("**/*.png")
    
    def __getitem__(self, index: int) -> torch.Tensor:
        data = torch.tensor([self.transform(Image.open(self.data[index + t])) for t in range(self.timestep)])
        return data
    
    def __len__(self):
        return len(self.data) - self.timestep
    
class ShuffledDataset(Dataset):
    def __init__(self, root: str, transform, timestep) -> None:
        super().__init__()
        self.transform = transform
        self.data = Path(root).glob("**/*.png")
        random.shuffle(self.data)
        self.timestep = timestep

    def __getitem__(self, index: int) -> torch.Tensor:
        data = torch.tensor([self.transform(Image.open(self.data[index + t])) for t in range(self.timestep)])
        return data
    
    def __len__(self):
        return len(self.data) - self.timestep
    

class Surgery_Dataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        super().__init__()
        self.transform = transform
        self.data = [str(p) for p in Path(root).glob("**/*.png")]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.transform(Image.open(self.data[index]))
        return data
    
    def __len__(self):
        return len(self.data)

    
if __name__ == "__main__":
    import torchvision.transforms as T
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])
    
    dataset = Surgery_Dataset(root="raw_data", transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True, drop_last=True)
    data = Surgery_Dataset(root="mask_data", transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, drop_last=True)

