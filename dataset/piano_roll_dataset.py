import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.sparse import csc_matrix
import json


class PianoRollDataset(Dataset):
    def __init__(self, args, split):
        self.data_dir = args.data_dir
        self.split = split
        self.file_paths = []
        with open('dataset/dataset.json', 'r') as f:
            self.genres = json.load(f)

        # Get paths of all files
        if self.split == "train":
            start_idx = 0
            end_idx = 100
        elif self.split == "val":
            start_idx = 100
            end_idx = 109
        elif self.split == "test":
            start_idx = 109
            end_idx = 117
        for i in range(start_idx, end_idx):
            root_path = os.path.join(self.data_dir, f"Piano_Rolls{i}")
            for midi_folder in os.listdir(root_path):
                self.file_paths.append(os.path.join(root_path, midi_folder))

        # Instruments in each channel
        self.instrument_indices = {
            'piano': 0,
            'piano1': 0,
            'a.piano': 0,
            'a.piano 1': 0,
            'a.piano 2': 0,
            'piano 1': 0,
            'piano 2': 0,
            'acoustic grand piano': 0,
            'organ': 0,
            'organ 3': 0,
            'piano (hi)': 0,
            'piano (lo)': 0,
            'synth': 0,
            'marimba': 0,
            'vibraphone': 0,
            'keyboard': 0,
            'e.piano 1': 0,
            'e.piano 2': 0,
            'grand piano': 0,

            'bass': 1,
            'acou bass': 1,
            'acoustic bass': 1,
            'fretless bass': 1,

            'strings': 2,
            'guitar': 2,
            'guitar 1': 2,
            'guitar 2': 2,
            'guitar 3': 2,
            'lead guitar': 2,
            'steel gtr': 2,
            'jazz gtr': 2,
            'harp': 2,
            'cello': 2,
            'violin': 2,
            'muted gtr': 2,
            'clean guitar': 2,
            'clean gtr': 2,
            'nylon gtr': 2,
            'acoustic guitar': 2,
            'banjo': 2,
            'viola': 2,

            'flute': 3,
            'clarinet': 3,
            'harmonica': 3,
            'oboe': 3,
            'piccolo': 3,
            'bassoon': 3,

            'trumpet': 4,
            'trombone': 4,
            'brass': 4,
            'brass 1': 4,
            'sax': 4,
            'alto sax': 4,
            'tenor sax': 4,
            'tuba': 4,
            'french horn': 4,
            'fr.horn': 4,
        }

        # Encoding for genres
        self.genre_indices = ["electronic", "pop", "classical", "rock"]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # return shape: (n_instruments, n_notes, total_duration)
        duration = 512#4096
        piano_roll = torch.zeros((5, 128, duration))
        
        file_path = self.file_paths[idx]

        # Get piano roll
        for file in os.listdir(file_path):
            if file[-4:] == '.npz':
                file_index = file.split('-')[0].split('.')[0]  # delete last part later
                instrument = file[:-4].split('-')[-1]
                data = np.load(os.path.join(file_path, file))
                single_piano_roll = csc_matrix((
                    data[f"pianoroll_0_csc_data"],
                    data[f"pianoroll_0_csc_indices"],
                    data[f"pianoroll_0_csc_indptr"]
                ), shape=data[f"pianoroll_0_csc_shape"]).toarray().T

                # Normalize
                single_piano_roll = single_piano_roll / 255

                # Padding / cropping
                num_repeats = duration // single_piano_roll.shape[1] + 1
                single_piano_roll = np.tile(single_piano_roll, (1, num_repeats))
                single_piano_roll = single_piano_roll[:, :duration]

                # Add instrument to the piano roll
                # If the same two instruments play the same note, 
                # use the one that is the loudest
                piano_roll[self.instrument_indices[instrument]] = np.maximum(
                    piano_roll[self.instrument_indices[instrument]], 
                    single_piano_roll
                )

        midi_name = os.path.basename(file_path)

        try:
            genre = self.genre_indices.index(self.genres[midi_name])
            return piano_roll, genre
        except KeyError:
            print(f"Warning: No genre found for {midi_name}, skipping")
        return self.getitem((idx + 1) % len(self))