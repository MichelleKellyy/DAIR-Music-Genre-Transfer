import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from miditok import REMI
from miditok.pytorch_data.datasets import DatasetMIDI

class REMIDataset(Dataset):
    def __init__(self, args, split):
        self.split = split

        # Get paths of all files
        self.file_paths = []
        if self.split == "train":
            start_idx = 1
            end_idx = 9
        elif self.split == "val":
            start_idx = 9
            end_idx = 11
        elif self.split == "test":
            start_idx = 11
            end_idx = 13
        for i in range(start_idx, end_idx):
            root_path = os.path.join(args.data_dir, f"folder_{i}")
            for midi_folder in os.listdir(root_path):
                self.file_paths.append(os.path.join(root_path, midi_folder))

        with open('dataset/dataset.json', 'r') as f:
            genres = json.load(f)

        tokenizer = REMI(params="C:/Users/Miche/Desktop/SMGT/tokenizer_12.json")

        self.dataset = DatasetMIDI(
            files_paths=self.file_paths,
            tokenizer=tokenizer,
            max_seq_len=1024,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            func_to_get_labels=self.get_label
        )

    def get_label(self, score, seq, file_path):
        with open('dataset/dataset.json', 'r') as f:
            genres = json.load(f)

        genre_indices = ["electronic", "pop", "classical", "rock"]
        return genre_indices.index(genres[os.path.basename(file_path)])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item["input_ids"][:5000], item["labels"]
