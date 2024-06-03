import pathlib
from typing import Sequence, Callable, List

import numpy as np
import jax.nn
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree
import numpyro.distributions as dist
import trapallada.util

class Encoder(eqx.Module):

    layers: list

    def __init__(self, input_size: tuple[int, int], dim_z: int, rng_key):

        # assert dim_z < 512, f'`dim_z` must be a dimension reduction for the last layer'

        key1, key2, key3, key4, key5, key6, key7, key8 = jr.split(rng_key, 8)

        self.layers = [
            eqx.nn.Conv2d(1, 10, kernel_size=4, stride=(1, 2), padding=1, key=key1),
            jax.nn.relu,
            eqx.nn.Conv2d(10, 10, kernel_size=4, stride=(2, 2), padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.Conv2d(10, 20, kernel_size=4, stride=(2, 2), padding=1, key=key3),
            jax.nn.relu,
            eqx.nn.Conv2d(20, 20, kernel_size=4, stride=(2, 2), padding=1, key=key4),
            jax.nn.relu,
            eqx.nn.Conv2d(20, 40, kernel_size=4, stride=1, padding=1, key=key5),
            jax.nn.relu,
            jnp.ravel,
        ]

        conv_layers_output_size = trapallada.util.size_across_conv_layers(input_size, trapallada.util.conv_output_size, self.layers)
        # print(conv_output_size)
        
        self.layers.extend([
            eqx.nn.Linear(conv_layers_output_size, 512, key=key6),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 2*dim_z, key=key7),
            lambda x: x.at[dim_z:].set(jax.nn.softplus(x[dim_z:]))
        ])

    def __call__(self, x: Float[Array, '1 height width'] ) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            x = layer(x)

        return x

    def debug_call(self, x: Float[Array, '1 height width'] ) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            x = layer(x)
            print(f'after {layer.__module__}: {x.shape}')


class Decoder(eqx.Module):

    layers: list

    def __init__(self, dim_z: int, conv_input_size: int, rng_key):

        key1, key2, key3, key4, key5, key6, key7, key8 = jr.split(rng_key, 8)

        self.layers = [
            eqx.nn.Linear(dim_z, 64, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(64, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, conv_input_size, key=key3),
            jax.nn.relu,
            lambda x: jnp.reshape(x, (40, 2, 14)), # from a 1D vector to a 3D array
            eqx.nn.ConvTranspose2d(40, 20, kernel_size=4, stride=1, padding=1, key=key4),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(20, 20, kernel_size=4, stride=(2, 2), padding=1, output_padding=(1, 1), key=key5),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(20, 10, kernel_size=4, stride=(2, 2), padding=1, key=key6),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(10, 10, kernel_size=4, stride=(2, 2), padding=1, output_padding=(1, 1), key=key7),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(10, 1, kernel_size=4, stride=(1, 2), padding=1, key=key8),
            # jax.nn.sigmoid
            jnp.tanh
        ]

    def __call__(self, x: Float[Array, 'dim_z'] ) -> Float[Array, '1 height width']:

        for layer in self.layers:

            x = layer(x)

        return x

    def debug_call(self, x: Float[Array, 'dim_z'] ) -> Float[Array, '1 height width']:

        for layer in self.layers:

            x = layer(x)
            print(f'after {layer.__module__}: {x.shape}')

def loss(
        model: eqx.Module,
        x: Float[Array, 'batch height width'],
        dim_z: int, x_sd: float, rng_key
        ) -> tuple[Float[Array, ''], tuple[jax.Array, jax.Array]]:

    # for the sake of readability
    encoder, decoder = model
    batch_size = len(x)

    z_mean_sd = jax.vmap(encoder)(x)

    # for the sake of readability
    z_mean, z_sd = z_mean_sd[:, :dim_z], z_mean_sd[:, dim_z:]

    # a sample for every element in the batch
    z = dist.Normal(loc=z_mean, scale=z_sd).sample(rng_key)
    
    # encoding
    x_pred = jax.vmap(decoder)(z)

    # reconstruction term
    loglikelihood = dist.Normal(loc=x_pred, scale=x_sd).log_prob(x).sum() / batch_size

    # KL divergence
    kl = -0.5 * jnp.sum(1. + 2.*jnp.log(z_sd) - z_mean**2 - z_sd**2) / batch_size

    return -loglikelihood + kl, (loglikelihood, kl)


def vae(input_size: int, dim_z: int, rng_key: jax.Array | int = jr.PRNGKey(42)):

    # if an `int` (as opposed to a jax `PRNGKey`) was passed...
    if isinstance(rng_key, int):
        rng_key = jr.PRNGKey(rng_key)
    
    encoder = Encoder(input_size, dim_z, rng_key)
    decoder = Decoder(
        dim_z,
        trapallada.util.size_across_conv_layers(input_size, trapallada.util.conv_output_size, encoder.layers),
        rng_key
    )

    return encoder, decoder

def load_vae(file: str | pathlib.Path, input_size: int, dim_z: int) -> eqx.Module:

    file = pathlib.Path(file)

    # check whether the file has the correct extension
    if file.suffix != '.eqx':
        raise ValueError("Invalid file extension. Expected '.eqx'.")

    # the random key is not gonna be used anyway
    model = vae(input_size, dim_z, jr.PRNGKey(42))

    with file.open('rb') as f:
        return eqx.tree_deserialise_leaves(f, model)

    return model

def save_vae(file: str | pathlib.Path, model: eqx.Module):
    
    with open(file, 'wb') as f:
        
        eqx.tree_serialise_leaves(f, model)
