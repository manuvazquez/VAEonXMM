from typing import Sequence, Callable, List

import jax.nn
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree
# import numpyro.distributions as dist
import trapallada.util

class Classifier(eqx.Module):

    layers: list

    def __init__(self, input_size: tuple, kernel_sizes: List[int], channels: List[int], stride: int, rng_key):

        padding = 0

        self.layers = []

        for in_channels, out_channels, key, kernel_size in zip(
            channels[:-1], channels[1:],
            jr.split(jr.PRNGKey(42), len(channels) - 1),
            kernel_sizes
        ):
            
            # the length of the feature map for the next layer
            input_size = trapallada.util.conv_output_size(input_size, (kernel_size,), (stride,), (padding,))
            print(f'{input_size=}')
            
            self.layers.extend([
                eqx.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, key=key),
                jax.nn.relu,
                eqx.nn.LayerNorm((out_channels, input_size[0]))
            ])

        # self.layers.append(jnp.squeeze)
        self.layers.append(jnp.ravel)

        key, subkey = jr.split(key)
        # self.layers.append(eqx.nn.Linear(in_features=channels[-1], out_features=3, key=subkey))
        print(channels[-1] * input_size[0])
        self.layers.append(eqx.nn.Linear(in_features=channels[-1] * input_size[0], out_features=3, key=subkey))
        
    def __call__(self, x: Float[Array, '1 length'] ) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            x = layer(x)

        return x

    def debug_call(self, x: Float[Array, '1 length'] ) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            x = layer(x)
            print(f'after {layer.__module__}: {x.shape}')