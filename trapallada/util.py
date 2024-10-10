from typing import Sequence, Callable, List
import inspect

import numpy as np
import pandas as pd
import equinox as eqx
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader

import jax
import jax.numpy as jnp
import jax.tree_util
from jaxtyping import PyTree

def conv_output_size(
        input_size: tuple | np.ndarray,
        filter_size: tuple | np.ndarray,
        stride: tuple | np.ndarray,
        padding: tuple | np.ndarray) -> tuple[int, int]:
    """
    Calculate the output size of a convolutional layer.
    
    Args:
        input_size (tuple | np.ndarray): The size of the input tensor.
        filter_size (tuple | np.ndarray): The size of the filter/kernel.
        stride (tuple | np.ndarray): The stride of the convolution operation.
        padding (tuple | np.ndarray): The amount of padding applied to the input.

    Returns:
        tuple: The output size of the convolutional layer.

    Examples:
        >>> conv_output_size((32, 32), (3, 3), (2, 2), (1, 1))
        (16, 16)
        >>> conv_output_size((32, 32), (3, 3), (1, 1), (0, 0))
        (30, 30)
    
    """

    return tuple(((np.array(input_size) - np.array(filter_size) + 2 * np.array(padding)) // np.array(stride)) + 1)

def transp_conv_output_size(
        input_size: tuple | np.ndarray,
        filter_size: tuple | np.ndarray,
        stride: tuple | np.ndarray,
        padding: tuple | np.ndarray) -> tuple[int, int]:
    """
    Calculate the output size of a transposed convolutional layer.

    Args:
        input_size (tuple | np.ndarray): The size of the input tensor.
        filter_size (tuple | np.ndarray): The size of the filter/kernel.
        stride (tuple | np.ndarray): The stride of the convolution operation.
        padding (tuple | np.ndarray): The amount of padding applied to the input.

    Returns:
        tuple[int, int]: The output size of the transposed convolutional layer.
    
    Examples:
        >>> transp_conv_output_size((32, 32), (3, 3), (2, 2), (1, 1))
        (63, 63)
        >>> transp_conv_output_size((32, 32), (3, 3), (1, 1), (0, 0))
        (34, 34)

    """

    return tuple((np.array(input_size) - 1) * np.array(stride) - 2 * np.array(padding) + np.array(filter_size))


def size_across_conv_layers(
    input_size: tuple[int, int],
    f: Callable[[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]], tuple[int, int]],
    layers: Sequence):

    size = input_size

    # for every layer...
    for l in layers:
        
        # ...if it is a convolutional layer...
        if isinstance(l, eqx.nn.Conv2d):
            
            size = f(size, l.kernel_size, l.stride, l.padding[0])
            
            last_conv_layer = l

    return np.array(size).prod() * last_conv_layer.out_channels


def make_results_df(n_epochs: int, metrics: List[str]=['loss', 'accuracy']) -> pd.DataFrame:
    """
    Create a DataFrame to store the results of training.

    Args:
        n_epochs (int): The number of epochs.

    Returns:
        pd.DataFrame: A DataFrame to store the results of training.

    Examples:
        >>> make_results_df(2)   
    """
    
    return pd.DataFrame({ (data, metric): [0.] * n_epochs for data in ['training', 'validation'] for metric in metrics})


def get_layers_n_parameters(model: PyTree) -> dict:
    """
    Get the number of parameters in every layer of a model.

    Args:
        model (PyTree): A model.

    Returns:
        dict: The number of parameters in each layer.
    """

    return {
        path_leave[0]: np.prod(path_leave[1].shape).item() 
        for path_leave in jax.tree_util.tree_leaves_with_path(model) if isinstance(path_leave[1], jax.Array)
    }

# TODO: this should be moved to the `analysis` module
def make_projections_df(ds: Dataset, dl: DataLoader, encoder: eqx.Module, n_projected_components: int = 2, random_state: int = 42) -> pd.DataFrame:
    """
    Build a `DataFrame` with the metadata and the projections of the latent space.

    Args:
        ds (Dataset): The full dataset.
        dl (DataLoader): A data loader to be processed.
        encoder (eqx.Module): The encoder.
        n_projected_components (int): The number of components for the t-SNE embeddings.

    Returns:
        A `DataFrame` with the metadata and the projections of the latent space.
    """
    
    # the dataset associated with `dl` is encoded, and the corresponding indexes (in the full datset) recorded
    z_s, i_s = [], []
    for i, x, _ in dl:
        z_s.append(jax.vmap(encoder)(x.numpy()))
        i_s.append(i)

    # for the sake of clarity (assuming `z` comprises mean and standard deviation)
    dim_z = z_s[-1].shape[-1] // 2

    # all the `z`s' means are (vertically) stacked together, and the standard deviations dropped
    z_s = jnp.vstack(z_s)[:, :dim_z]

    # t-SNE is applied to the (means of the) `z`s
    t_sne_embeddings = TSNE(n_components=n_projected_components, perplexity=30, random_state=random_state).fit_transform(z_s)

    # names for columns
    z_s_names = [f'z_{i}' for i in range(1, 1 + dim_z)]
    t_sne_names = [f'tsne_{i}' for i in range(1, 1 + n_projected_components)]

    # rows are selected from the dataset's metadata
    metadata_df = ds.metadata_df.loc[torch.hstack(i_s)]

    # a `DataFrame` with the `z`s and the t-SNE embeddings (indexed just like thea above `DataFrame`)
    z_tsne_df = pd.DataFrame(np.hstack((z_s, t_sne_embeddings)), columns=z_s_names + t_sne_names, index=metadata_df.index)

    return pd.concat([metadata_df, z_tsne_df], axis=1)

def func_args_with_defaults(func):
    """
    Returns a dictionary of arguments and their default values for a given function.

    Args:
        func (function): The function to inspect.

    Returns:
        dict: A dictionary where keys are argument names and values are their default values.
              If an argument has no default value, it is set to None.
    """
    signature = inspect.signature(func)
    
    return {
        param.name: param.default if param.default is not param.empty else None
        for param in signature.parameters.values()
    }

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    
    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for recursion (used internally).
        sep (str): The separator between parent and child keys.
        
    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)