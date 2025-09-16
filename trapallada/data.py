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
    
# TODO: this is a dataset of *windows*, not of *signals* (the appropriate files should also be renamed)

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
        i_to_validation: np.ndarray | None = None
        ) -> tuple[DataLoader, DataLoader] | tuple[tuple[DataLoader, DataLoader], tuple[Dataset, Dataset]]:
    
    # for the sake of readability
    all_indexes = np.arange(len(ds))

    train_idx, validation_idx = train_test_split(
        all_indexes, 
        **splitting_parameters, 
        shuffle=True,
    )

    # if indexes to be included in the validation set are provided...
    if i_to_validation is not None:
        
        # ...those that ended up in the training set...
        intersection = np.intersect1d(i_to_validation, train_idx)

        # ...must be replaced with randomly chosen indexes from the validation set (excluding the ones provided)
        replacements = np.random.RandomState(batching_random_state).choice(
            np.setdiff1d(validation_idx, i_to_validation), size=len(intersection), replace=False
            )
        # replacements = np.random.RandomState(batching_random_state).choice(
        #     validation_idx, size=len(intersection), replace=False
        #     )
        
        # the training and validation set indexes are updated
        train_idx = np.concatenate((np.setdiff1d(train_idx, intersection), replacements))
        validation_idx = np.concatenate((np.setdiff1d(validation_idx, replacements), intersection))

        assert (np.union1d(train_idx, validation_idx) == all_indexes).all()
        assert np.intersect1d(train_idx, validation_idx).size == 0

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

def sample_dataset_parameters(subdir='rates_large_infs', include_parameters: bool = False) -> dict[str, pathlib.Path]:
    
    base_dir = trapallada.util.get_base_directory()

    # basedir = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'rates_large_infs'
    # basedir = trapallada.util.get_base_directory() / 'data' / 'rates_large_infs'
    # basedir = trapallada.util.get_base_directory() / 'data' / 'mini_channels'
    data_dir = base_dir / 'data' / subdir

    assert data_dir.exists()

    res = dict(
        signals_filename=data_dir / 'signals.pt',
        masks_filename=data_dir / 'masks.pt',
        metadata_filename=data_dir / 'metadata.parquet',
        )
    
    if include_parameters:
        
        res['data_preprocessing_parameters_filename'] = data_dir / 'data_preprocessing_parameters.yaml'

    for file in res.values():
        
        assert file.exists(), f'"{file}" does not exist'
    
    return res

class Labels:

    def __init__(self, reference_signals_file: str | pathlib.Path | None = None, annotated_windows_file: str | pathlib.Path | None = None):

        self.signals: list[str] = []

        # if reference_signals_file is None:
            
        #     self.reference_signals_file = trapallada.util.get_base_directory() / 'interesting.yaml'

        # else:
            
        #     # just in case a `str` was passed...
        #     self.reference_signals_file = pathlib.Path(reference_signals_file)

        self.reference_signals_data: dict[str, list[str]] | None = None
        
        if reference_signals_file is not None:

            with open(reference_signals_file) as f:

                self.reference_signals_data = yaml.safe_load(f)

            self.signals += list(self.reference_signals_data['signals of interest'].keys())

        # if a file with annotated windows is provided...
        if annotated_windows_file is not None:
            
            self.annotated_windows_file = pd.read_parquet(annotated_windows_file)
            
            self.signals += self.annotated_windows_file['signal'].unique().tolist()
        
    # @property
    # def signals(self):
        
    #     return list(self.reference_signals_data['signals of interest'].keys())
    
    @property
    def signals_without_prefix(self):
        
        return [e.split('/')[1 ]for e in self.signals]
    
    # @property
    # def signals_to_labels(self):
        
    #     return self.reference_signals_data['signals of interest']
    
    @property
    def reference_signals_and_labels(self):

        if self.reference_signals_data is None:
            return []
        else:
            return list(map(list, self.reference_signals_data['signals of interest'].items()))
    
    def __str__(self):

        res = 'Reference signals:\n'    

        if self.reference_signals_data is not None:
            res += pd.DataFrame(self.reference_signals_data['signals of interest'].items(), columns=['signal', 'alias']).to_string()

        return res

    def __repr__(self):
        
        return self.__str__()

def labels_files() -> dict[str, pathlib.Path]:
    
    base_dir = trapallada.util.get_base_directory()

    res = dict(
        reference_signals_file=base_dir / 'interesting.yaml',
        annotated_windows_file=base_dir / 'annotations' / 'annotated_windows.parquet',
        )

    for file in res.values():
        
        assert file.exists(), f'"{file}" does not exist'
    
    return res