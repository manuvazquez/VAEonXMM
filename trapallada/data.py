import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, Float

import trapallada.logging as log

logger = log.setup_logging(__name__)
    
class SignalsDataset(Dataset):

    def __init__(
            self,
            signals_filename: str | pathlib.Path,
            metadata_filename: str | pathlib.Path,
            masks_filename: str | pathlib.Path,
            normalize: bool = False, fft: bool = False
        ):

        self.signals: Float[Array, 'signal channel length'] = torch.load(signals_filename, weights_only=False)
        self.masks: Float[Array, 'signal length'] = torch.load(masks_filename, weights_only=False)

        # TODO: at some point the type of `self.signals` should be guaranteed to be `np.ndarray`, never `torch.Tensor`
        if type(self.signals) == np.ndarray:
            
            self.signals = torch.from_numpy(self.signals)

        # TO-DO: this path is not longer used in any experiment
        if normalize:
            
            mean = self.signals.mean(axis=-1)
            std = self.signals.std(axis=-1)
            
            self.signals = (self.signals - mean[..., None]) / std[..., None]

            # TODO: compute `i_constant` first, and filter out the signals

            # indexes of the constant signals
            # CAVEAT: this relies exclusively on the *first* channel of each signal
            i_constant = torch.nonzero(std == 0.)[:, 0]

            # the number of them
            n_constant_signals = len(i_constant)

            # a mask to keep only the non-constant signals
            to_keep = torch.full((len(self.signals),), 1, dtype=bool)
            to_keep[i_constant] = False

            # only non-constant signals are kept
            self.signals = self.signals[to_keep]

            logger.info('normalizing signals...')

            if n_constant_signals > 0:

                logger.warning(f'dismissed {n_constant_signals} constant signals (that cannot be normalized); {len(self.signals)} left')

        # TODO: no need...
        # factored out code to initialize `labels` and `labels_names` when `metadata_filename`
        # is not provided or doesn't contain labels
        make_labels_and_labels_names = lambda : (np.full((len(self.signals),), np.nan), {})

        # metadata is read
        self.metadata_df = pd.read_parquet(metadata_filename)

        if normalize:
        
            # only metadata for non-constant signals is kept
            self.metadata_df = self.metadata_df.drop(i_constant).reset_index(drop=True)

        if 'label' in self.metadata_df.columns:

            self.labels = self.metadata_df['label'].cat.codes
            self.labels_names = dict(enumerate(self.metadata_df['label'].cat.categories))

        else:
                
            self.labels, self.labels_names = make_labels_and_labels_names()

        if fft:

            logger.info('computing the FFT of each signal...')
        
            self.signals = torch.abs(torch.fft.rfft(self.signals, norm='forward'))

            # --------------------- "normalization" ---------------------

            # # meand and standard deviation separately *for every channel*
            # mean = self.signals.mean(axis=(0, -1))
            # std = self.signals.std(axis=(0, -1))

            # self.signals = (self.signals - mean[None, :, None]) / std[None, :, None]

            # --------------------- "normalization" ---------------------

            logger.info(f'dismissing the right half of each FFT (length is now {self.signals.shape[-1]})...')


    def __len__(self):

        return len(self.signals)

    def __getitem__(self, idx):
        
        return idx, self.signals[idx], self.masks[idx], self.labels[idx]
    
def dataloaders_from_dataset(
        ds: Dataset, splitting_parameters: dict[str, float], batch_size: int, batching_random_state: int, return_datasets: bool = False
        ) -> tuple[DataLoader, DataLoader] | tuple[tuple[DataLoader, DataLoader], tuple[Dataset, Dataset]]:

    train_idx, validation_idx = train_test_split(
        np.arange(len(ds)), 
        **splitting_parameters, 
        shuffle=True,
        # stratify=ds.labels
        stratify=ds.labels if not np.isnan(ds.labels).all().item() else None
    )

    train_ds, validation_ds = Subset(ds, train_idx), Subset(ds, validation_idx)

    # a random key...
    key, subkey = jr.split(jr.PRNGKey(batching_random_state))

    # ...to compute a seed for a PyTorch `Generator`...
    torch_seed = jr.bits(subkey, (1,), dtype=jnp.uint32).item()
    torch_rng = torch.Generator().manual_seed(torch_seed)

    # ...meant for "batching" in the `DataLoader`'s
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, generator=torch_rng)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, generator=torch_rng)

    if return_datasets:
        
        return (train_dl, validation_dl), (train_ds, validation_ds)

    return train_dl, validation_dl