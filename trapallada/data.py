import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import jax.random as jr
import jax.numpy as jnp

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

    def __init__(self, signals_filename: str | pathlib.Path, metadata_filename: str | pathlib.Path | None, normalize: bool = False):

        self.signals = torch.load(signals_filename)

        # n_signals = len(self.signals)

        if normalize:
            mean = self.signals.mean(axis=-1)
            std = self.signals.std(axis=-1)
            
            self.signals = (self.signals - mean[..., None]) / std[..., None]

            # indexes of the constant signals
            i_constant = torch.nonzero(std == 0.)[:, 0]

            # a mask to keep only the non-constant signals
            to_keep = torch.full((len(self.signals),), 1, dtype=bool)
            to_keep[i_constant] = False

            # only non-constant signals are kept
            self.signals = self.signals[to_keep]

            if torch.isnan(self.signals).any().item():

                print('oops normalizing signals produced NaNs')
        
        # if metadata_filename:
        #     self.metadata_df = pd.read_parquet(metadata_filename)
        #     self.labels = self.metadata_df['label'].cat.codes
        #     self.labels_names = dict(enumerate(self.metadata_df['label'].cat.categories))
        # else:
        #     self.labels = torch.full((len(self.signals),), torch.nan)
        #     self.labels_names = {}

        # factored out code to initialize `labels` and `labels_names` when `metadata_filename`
        # is not provided or it doesn't contain labels
        # make_labels_and_labels_names = lambda : (torch.full((len(self.signals),), torch.nan), {})
        make_labels_and_labels_names = lambda : (np.full((len(self.signals),), np.nan), {})

        if metadata_filename:
            
            self.metadata_df = pd.read_parquet(metadata_filename)

            # only metadata for non-constant signals is kept
            self.metadata_df = self.metadata_df.drop(i_constant)
            # self.metadata_df = self.metadata_df.iloc[to_keep]

            if 'label' in self.metadata_df.columns:

                self.labels = self.metadata_df['label'].cat.codes
                self.labels_names = dict(enumerate(self.metadata_df['label'].cat.categories))

            else:
                    
                self.labels, self.labels_names = make_labels_and_labels_names()
        else:
            
            self.labels, self.labels_names = make_labels_and_labels_names()

    def __len__(self):

        return len(self.signals)

    def __getitem__(self, idx):
        
        return idx, self.signals[idx], self.labels[idx]
    
def dataloaders_from_dataset(
        ds: Dataset, splitting_parameters: dict[str, float], batch_size: int, batching_random_state: int
        ) -> tuple[DataLoader, DataLoader]:

    train_idx, validation_idx = train_test_split(
        np.arange(len(ds)), 
        **splitting_parameters, 
        shuffle=True,
        # stratify=ds.labels
        stratify=ds.labels if not np.isnan(ds.labels).all().item() else None
    )

    train_ds, validation_ds = Subset(ds, train_idx), Subset(ds, validation_idx)

    # a random key...
    # key, subkey = jr.split(rng_key)
    key, subkey = jr.split(jr.PRNGKey(batching_random_state))

    # ...to compute a seed for a PyTorch `Generator`...
    torch_seed = jr.bits(subkey, (1,), dtype=jnp.uint32).item()
    torch_rng = torch.Generator().manual_seed(torch_seed)

    # ...meant for "batching" in the `DataLoader`'s
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, generator=torch_rng)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, generator=torch_rng)

    return train_dl, validation_dl