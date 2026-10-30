import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from dataloader.custom_dataset import CustomDataset

import os


class DataLoader():
    dataset_source_folder_path = None
    batch_size = None

    def __init__(self, dPath, batch_size=1):
        self.dataset_source_folder_path = dPath
        self.batch_size = batch_size

    # Used for standardizing input dimensions without
    def depth_collate_fn(self, data):
        rgb_images, depth_maps = zip(*data)

        max_h = max(img.shape[-2] for img in rgb_images)
        max_w = max(img.shape[-1] for img in rgb_images)

        target_h = ((max_h + 31) // 32) * 32
        target_w = ((max_w + 31) // 32) * 32

        def pad_image(image, target_h, target_w, value=0.0):
            h, w = image.shape[-2:]

            padding = (
                0, target_w - w,  # Left, right
                0, target_h - h  # Top, bottom
            )

            return F.pad(image, padding, value=value)

        rgb_batch = []
        depth_batch = []
        masks = []

        for rgb, depth in zip(rgb_images, depth_maps):

            # Verify RGB and depth are spatially aligned
            if rgb.shape[-2:] != depth.shape[-2:]:
                raise ValueError("RGB and depth dimensions do not match")

            h, w = rgb.shape[-2:]

            # Pad RGB image (0 = ImageNet mean after normalization)
            rgb = pad_image(rgb, target_h, target_w)
            rgb_batch.append(rgb)

            # Pad depth map
            depth = pad_image(depth, target_h, target_w)
            depth_batch.append(depth)

            # Create mask: 1 = original pixels, 0 = padding
            mask = torch.zeros(
                (1, target_h, target_w),
                dtype=torch.float32
            )

            mask[:, :h, :w] = 1.0
            masks.append(mask)

        # Stack individual samples into batches
        rgb_batch = torch.stack(rgb_batch)
        depth_batch = torch.stack(depth_batch)
        mask_batch = torch.stack(masks)

        return rgb_batch, depth_batch, mask_batch


    def load(self):
        train_path = self.dataset_source_folder_path / "train"
        test_path = self.dataset_source_folder_path / "test"

        print(f"Loading data...")

        # Retrieves a list of strings of image file names
        train_list = [d.name for d in os.scandir(train_path / "rgb") if d.is_file()]
        test_list = [d.name for d in os.scandir(train_path / "rgb") if d.is_file()]


        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        depth_transform = transforms.Compose([
            transforms.ToTensor()
        ])


        trainset = CustomDataset(train_list, train_path, rgb_transform=rgb_transform, depth_transform=depth_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, collate_fn=self.depth_collate_fn, num_workers=0)

        testset = CustomDataset(test_list, test_path, rgb_transform=rgb_transform, depth_transform=depth_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, collate_fn=self.depth_collate_fn, shuffle=False, num_workers=0)


        return trainloader, testloader



