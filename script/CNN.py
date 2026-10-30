# Script to run the model

from dataloader.dataloader import DataLoader
from model.model import Model

from pathlib import Path

# Datalaoder initialization
data_path = Path(__file__).resolve().parent.parent / "data"

trainloader, testloader = DataLoader(data_path, batch_size=32).load()


# Object initialization

image_batch, depth_batch, mask_batch = next(iter(trainloader))
# print(first_batch.shape)    # Shape of entire batch
print(image_batch.shape) # Shape of first image tensor
print(depth_batch.shape) # Shape of first depth tensor
print(mask_batch.shape)


# print(depth_trainset[0].shape, depth_testset[0].shape)
# Running the model
