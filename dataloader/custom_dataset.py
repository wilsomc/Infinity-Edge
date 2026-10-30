'''
    Custom dataset for torch.vision iterator
    Processes rgb images and gt depth maps
'''

import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_list, dir_path, rgb_transform=None, depth_transform=None):
        self.dir_path = dir_path

        self.image_list = image_list
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.dir_path / "rgb"
        depth_path = self.dir_path / "depth_bfx"

        file_prefix = Path(self.image_list[idx]).stem
        image_file_path = Path(image_path / file_prefix)
        depth_file_path = Path(depth_path / file_prefix)

        rgb_image = Image.open(image_file_path.with_suffix(".jpg")).convert("RGB")
        depth_image = Image.open(depth_file_path.with_suffix(".png"))

        # Apply transformations (like converting to a PyTorch Tensor)
        if self.rgb_transform:
            rgb_tensor = self.rgb_transform(rgb_image)
        else:
            # Fallback if no transform is provided (less ideal)
            rgb_tensor = torch.tensor(rgb_image, dtype=torch.float32)

        if self.depth_transform:
            depth_tensor = self.depth_transform(depth_image)
        else:
            depth_tensor = torch.tensor(depth_image, dtype=torch.float32)


        return rgb_tensor, depth_tensor




