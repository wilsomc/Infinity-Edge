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
        image_file_path = image_path / f"{file_prefix}.jpg"
        depth_file_path = depth_path / f"{file_prefix}.png"

        try:
            with Image.open(image_file_path) as img:
                img.load()
                rgb_image = img.convert("RGB")

            with Image.open(depth_file_path) as img:
                img.load()
                depth_image = img.copy()

        except Exception as e:
            print(f"Error: {e}")
            print(f"Attempting to load image: {file_prefix}")
            print(f"Image Path: {image_file_path}")
            print(f"Depth Path: {depth_file_path}")

            raise RuntimeError(
                f"Failed to load dataset index {idx}"
            ) from e

        # Apply transformations (like converting to a PyTorch Tensor)
        if self.rgb_transform:
            rgb_tensor = self.rgb_transform(rgb_image)
        else:
            raise ValueError("RGB transform is required")

        if self.depth_transform:
            depth_tensor = self.depth_transform(depth_image)
        else:
            raise ValueError("Depth transform is required")


        return rgb_tensor, depth_tensor




