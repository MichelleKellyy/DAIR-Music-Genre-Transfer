import torch
from torch.utils.data import Dataset
import os


class PianoRollDataset(Dataset):
    def __init__(self, args, split):
        self.data_dir = args.data_dir
        self.split = split
        self.file_paths = ["C:/Users/Miche/Downloads/Piano_Rolls0/Piano_Rolls0"]

        # Get paths of all training files
        for path, _, files in os.walk(os.path.join(self.data_dir, self.split)):
            for file in files:
                self.file_paths.append(os.path.join(self.data_dir, self.split, path, file))

    def __len__(self):
        #return len(self.file_paths)
        return 1000

    def __getitem__(self, idx):
        # return shape: (n_instruments, n_notes, total_duration)
        x = 0.7 * torch.ones((8, 30, 500))
        return x
