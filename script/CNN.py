# Script to run the model

from dataloader.dataloader import DataLoader
from model.model import Model

from pathlib import Path
data_path = Path(__file__).resolve().parent / "data"

# Object initialization
dataloader = DataLoader(data_path, "train", "test")
cnn_model = Model()


trainset, testset = dataloader.load()


# Running the model
