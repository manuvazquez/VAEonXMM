import torch
import equinox as eqx
import pandas as pd
from jaxtyping import Array, Float

import matplotlib.pyplot as plt
import altair as alt

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

def signals(signals: Float[Array, 'batch 1 length'], metadata_df: pd.DataFrame, n_samples: int = 100):

    df = pd.DataFrame(signals.squeeze().T, columns=metadata_df['file name'].str.extract(r'.*/(.*).(?:jpg|mat)')[0])
    df.columns = df.columns.str.replace('.', '')
    
    chart = alt.Chart(
        df.iloc[:n_samples].reset_index()
    ).transform_fold(
        df.columns.to_list()
    ).mark_line().encode(
        x='index:Q', y='value:Q', color='key:N'
    ).properties(width=1_200, height=400)

    return chart