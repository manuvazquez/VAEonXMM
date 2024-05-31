import pathlib

import pandas as pd
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, images_filename: str | pathlib.Path, labels_filename: str | pathlib.Path):

        self.imgs = torch.load(images_filename)
        
        self.df = pd.read_parquet(labels_filename)
        self.labels = self.df['label'].cat.codes
        self.labels_names = dict(enumerate(self.df['label'].cat.categories))

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, idx):
        
        #  `Tensor`s are *normalized* so that numbers are spread between $-1$ and $1$ (rather than between $0$ and $1$).
        # TODO: use *torch* machinery for normalization purposes?
        # return 2*(self.imgs[idx] - 0.5), (self.labels[idx], self.df.iloc[idx]['std'])
        return idx, 2*(self.imgs[idx] - 0.5), self.labels[idx]
    
class SignalsDataset(Dataset):

    def __init__(self, signals_filename: str | pathlib.Path, metadata_filename: str | pathlib.Path):

        self.signals = torch.load(signals_filename)
        
        self.metadata_df = pd.read_parquet(metadata_filename)
        self.labels = self.metadata_df['label'].cat.codes
        self.labels_names = dict(enumerate(self.metadata_df['label'].cat.categories))

    def __len__(self):

        return len(self.signals)

    def __getitem__(self, idx):
        
        return idx, self.signals[idx], self.labels[idx]