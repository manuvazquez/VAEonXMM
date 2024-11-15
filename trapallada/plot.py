from typing import Sequence

import torch
import equinox as eqx
import pandas as pd
from jaxtyping import Array, Float, PRNGKeyArray
import jax.random as jr

import matplotlib.pyplot as plt
import altair as alt

import trapallada.data

def img_and_pred(model: tuple[eqx.Module, eqx.Module], x: torch.Tensor, dim_z: int):
    """
    A function to plot an image and its decoded counterpart side by side.

    Args:
        model (tuple): A tuple containing the encoder and decoder models.
        x (torch.Tensor): The input image tensor.
        dim_z (int): The dimension of the latent space.
    """
    
    # axes
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # original image
    axes[0].imshow(x[0].numpy())

    # for the sake of readability
    encoder, decoder = model
    
    # decoded image
    axes[1].imshow(decoder(encoder(x.numpy())[:dim_z])[0])

# def signals_dataset(
#         ds: trapallada.data.SignalsDataset, columns: Sequence[str] | None = None,
#         n_signals: int | None = 15, n_samples: int = 1_000, rng_key: PRNGKeyArray = jr.PRNGKey(42)):
#     """
#     Generate a chart of signals from a dataset.

#     Args:
#         ds (trapallada.data.SignalsDataset): The dataset containing the signals.
#         columns (Sequence[str] | None, optional): The names of the columns. If `None`, they are inferred.
#         n_signals (int | None, optional): The number of signals to include in the chart. If None, all signals in the dataset will be included. Defaults to 15.
#         n_samples (int, optional): The number of samples to include in the chart. Defaults to 1000.
#         rng_key (PRNGKeyArray, optional): The random number generator key. Defaults to jr.PRNGKey(42).

#     Returns:
#         alt.Chart: The chart of signals.
#     """
    
#     if n_signals is None:
#         n_signals = len(ds)

#     i_signals = jr.choice(rng_key, len(ds), (n_signals, ), replace=False).tolist()

#     if columns is None:
#         # columns = ds.metadata_df.iloc[i_signals]['file name'].str.extract(r'.*/(.*).(?:jpg|mat)')[0]
#         columns = ds.metadata_df.iloc[i_signals]['file name'].str.extract(r'.*/(.*).(?:jpg|mat)')[0].str.replace('.', '')
#     else:
#         assert len(columns) == n_signals, 'The number of columns must match the number of signals.'

#     df = pd.DataFrame(
#         ds.signals[i_signals].squeeze().T,
#         # columns=ds.metadata_df.iloc[i_signals]['file name'].str.extract(r'.*/(.*).(?:jpg|mat)')[0]
#         columns=columns
#     )
    
#     chart = alt.Chart(
#         df.iloc[:n_samples].reset_index()
#     ).transform_fold(
#         df.columns.to_list()
#     ).mark_line().encode(
#         x='index:Q', y='value:Q', color='key:N'
#     ).properties(width=1_200, height=400)

#     return chart

def signal(signal: Float[Array, 'length']):

    return alt.Chart(
        pd.DataFrame({'y': signal}).reset_index(names='x')
    ).mark_line().encode(
        x='x', y='y'
    ).properties(width=1_200, height=400)

def df(df: pd.DataFrame, width: int = 1_200, height: int = 400, columns: Sequence[str] | None = None, index_type: str = 'Q'):

    if columns is None:
        columns = df.columns.to_list()
    else:
        assert len(columns) == df.shape[1], 'The number of columns must match that in the DataFrame.'

    chart = alt.Chart(
        df.rename(columns=dict(zip(df.columns, columns))).reset_index(names='index')
    ).transform_fold(
        columns
    ).mark_line().encode(
        x=f'index:{index_type}', y='value:Q', color='key:N'
    ).properties(width=width, height=height)

    return chart

def signals_array(signals: Float[Array, 'signal length'], columns: Sequence[str] | None = None):
    """
    A function to plot a set of signals.

    Args:
        signals (Array): A set of signals.
        columns (Sequence[str] | None, optional): The names of the columns. If `None`, they are inferred. Defaults to None.
    """

    if columns is None:
        columns = [f's_{i}' for i in range(len(signals))]
    else:
        assert len(columns) == len(signals), 'The number of columns must match the number of signals.'

    return df(pd.DataFrame(signals.T, columns=columns))