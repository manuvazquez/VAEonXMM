import pathlib

import pandas as pd
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, images_filename: str | pathlib.Path, labels_filename: str | pathlib.Path):

        self.imgs = torch.load(images_filename)
        
        self.labels_df = pd.read_parquet(labels_filename)
        self.labels = self.labels_df['label'].cat.codes
        self.labels_names = dict(enumerate(self.labels_df['label'].cat.categories))

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, idx):
        
        return 2*(self.imgs[idx] - 0.5), (self.labels[idx], self.labels_df.iloc[idx]['std'])