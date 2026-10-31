# Script to run the model

import torch
from dataloader.dataloader import DataLoader
from model.model import Model

from matplotlib import pyplot as plt
from pathlib import Path

# Select training device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)

# Datalaoder instantiation
data_path = Path(__file__).resolve().parent.parent / "data"

trainloader, testloader = DataLoader(data_path, batch_size=4).load()

# Model instantiation
cnn = Model(trainloader, testloader)

cnn.run(device)

# Object instantiation




# Running the model


image_batch, depth_batch, mask_batch = next(iter(trainloader))
# print(first_batch.shape)    # Shape of entire batch
print(image_batch.shape) # Shape of first image tensor
print(depth_batch.shape) # Shape of first depth tensor
print(mask_batch.shape)


print("--------Starting CNN Model---------")
# cnn.run()
