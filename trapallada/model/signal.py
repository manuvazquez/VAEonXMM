from typing import Sequence, Callable, List

import jax.nn
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray
# import numpyro.distributions as dist
import trapallada.util

class Classifier(eqx.Module):

    layers: list

    def __init__(self, input_size: int, n_classes: int, kernel_sizes: List[int], channels: List[int], stride: int, dropout_rate: float, key: PRNGKeyArray):

        padding = 0

        self.layers = []

        conv_output_len = input_size

        for in_channels, out_channels, key, kernel_size in zip(
            channels[:-1], channels[1:],
            jr.split(key, len(channels) - 1),
            kernel_sizes
        ):
            
            # the length of the feature map for the next layer
            conv_output_len = trapallada.util.conv_output_size(conv_output_len, (kernel_size,), (stride,), (padding,))
            print(f'{conv_output_len=}')

            key, subkey = jr.split(key)
            
            self.layers.extend([
                eqx.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, key=subkey),
                jax.nn.relu,
                eqx.nn.LayerNorm((out_channels, conv_output_len[0]))
            ])

            self.layers.append(eqx.nn.Dropout(p=dropout_rate))

        self.layers.append(jnp.ravel)

        key, subkey = jr.split(key)
        print(out_channels * conv_output_len[0])
        self.layers.append(eqx.nn.Linear(in_features=out_channels * conv_output_len[0], out_features=n_classes, key=subkey))

    def __call__(self, x: Float[Array, '1 length'], key: PRNGKeyArray) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            if isinstance(layer, eqx.nn.Dropout):
                
                x = layer(x, key=key)

            else:

                x = layer(x)

        return x

    def debug_call(self, x: Float[Array, '1 length'], key: PRNGKeyArray) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            if isinstance(layer, eqx.nn.Dropout):
                
                x = layer(x, key=key)

            else:

                x = layer(x)

            print(f'after {layer.__module__}: {x.shape}')