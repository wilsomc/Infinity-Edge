import torch
from torch.utils.data import Dataset
from torchvision import datasets

class DataLoader:
    dataset_source_folder_path = None
    dataset_train_source_file_name = None
    dataset_test_source_file_name = None

    def __init__(self, dPath, dTrainFileName, dTestFileName):
        self.dataset_source_folder_path = dPath
        self.dataset_train_source_file_name = dTrainFileName
        self.dataset_test_source_file_name = dTestFileName

    def load(self):
        print(f"Loading data...")
        return



