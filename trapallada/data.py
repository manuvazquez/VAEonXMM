import pathlib

import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, Float

import trapallada.logging as log
import trapallada.util

logger = log.setup_logging(__name__)
    
class SignalsDataset(Dataset):

    def __init__(
            self,
            signals_filename: str | pathlib.Path,
            metadata_filename: str | pathlib.Path,
            masks_filename: str | pathlib.Path,
            fft: bool = False
        ):

        self.signals: Float[Array, 'signal channel length'] = torch.load(signals_filename, weights_only=False)
        self.masks: Float[Array, 'signal length'] = torch.load(masks_filename, weights_only=False)

        self.windows_length = self.signals.shape[-1]

        # TODO: at some point the type of `self.signals` should be guaranteed to be `np.ndarray`, never `torch.Tensor`
        if type(self.signals) == np.ndarray:
            
            self.signals = torch.from_numpy(self.signals)

        # metadata is read
        self.metadata_df = pd.read_parquet(metadata_filename)

        if fft:

            logger.info('computing the FFT of each signal...')
        
            self.signals = torch.abs(torch.fft.rfft(self.signals, norm='forward'))

            # --------------------- "normalization" ---------------------

            # # meand and standard deviation separately *for every channel*
            # mean = self.signals.mean(axis=(0, -1))
            # std = self.signals.std(axis=(0, -1))

            # self.signals = (self.signals - mean[None, :, None]) / std[None, :, None]

            # --------------------- "noitazilamron" ---------------------

            logger.info(f'dismissing the right half of each FFT (length is now {self.signals.shape[-1]})...')


    def __len__(self):

        return len(self.signals)

    def __getitem__(self, idx):
        
        return idx, self.signals[idx], self.masks[idx]
    
def dataloaders_from_dataset(
        ds: Dataset, splitting_parameters: dict[str, float], batch_size: int, batching_random_state: int, return_datasets: bool = False,
        to_validation: list[str] | None = None
        ) -> tuple[DataLoader, DataLoader] | tuple[tuple[DataLoader, DataLoader], tuple[Dataset, Dataset]]:
    
    # len_ds = len(ds)

    # if to_validation:
    #     len(to_validation)
        
    train_idx, validation_idx = train_test_split(
        np.arange(len(ds)), 
        **splitting_parameters, 
        shuffle=True,
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

def sample_dataset_parameters():

    basedir = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'rates_large_infs'

    assert basedir.exists()

    res = dict(
        signals_filename=basedir / 'signals.pt',
        metadata_filename=basedir / 'metadata.parquet',
        masks_filename=basedir / 'masks.pt'
        )

    for file in res.values():
        assert file.exists(), f'"{file}" does not exist'
    
    return res

class Labels:

    def __init__(self, data_file: str | pathlib.Path | None = None):

        if data_file is None:
            
            self.data_file = trapallada.util.get_base_directory() / 'interesting.yaml'

        else:
            
            # just in case a `str` was passed...
            self.data_file = pathlib.Path(data_file)
        
        with self.data_file.open() as f:

            self.data = yaml.safe_load(f)
        
    @property
    def signals(self):
        
        return list(self.data['signals of interest'].keys())
    
    @property
    def signals_without_prefix(self):
        
        return [e.split('/')[1 ]for e in self.signals]
    
    @property
    def signals_to_labels(self):
        
        return self.data['signals of interest']
    
    @property
    def signals_and_labels(self):
        
        return list(map(list, self.data['signals of interest'].items()))
    
    def __str__(self):
        
        return pd.DataFrame(self.signals_to_labels.items(), columns=['signal', 'alias']).to_string()
    
    def __repr__(self):
        
        return self.__str__()
