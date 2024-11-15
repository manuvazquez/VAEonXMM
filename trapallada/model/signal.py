import pathlib
from typing import Sequence, List

import yaml

import jax.nn
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PyTree, PRNGKeyArray
import numpyro.distributions as dist
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

            key, subkey = jr.split(key)
            
            self.layers.extend([
                eqx.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, key=subkey),
                jax.nn.relu,
                eqx.nn.LayerNorm((out_channels, conv_output_len[0]))
            ])

            self.layers.append(eqx.nn.Dropout(p=dropout_rate))

        self.layers.append(jnp.ravel)

        key, subkey = jr.split(key)

        # after `raveling`, the output from the previous layer (now `in_features`) is flattened
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

class Encoder(eqx.Module):

    layers: list
    layers_spatial_sizes: List[int]

    def __init__(self, input_size: Sequence[int], dim_z: int, kernel_sizes: List[int], channels: List[int], stride: int, key: PRNGKeyArray):

        # for the sake of readability
        n_conv_layers = len(channels)
        
        assert len(kernel_sizes) == n_conv_layers, f'number of kernel sizes ({len(kernel_sizes)}) must match number of channels ({n_conv_layers})'

        padding = 0

        # for the sake of readability
        n_input_maps, input_maps_len = input_size

        # the *first* number of input channels is determined by the data
        channels = [n_input_maps] + channels

        self.layers = []

        # the initial length of the feature map
        conv_output_len = input_maps_len

        # it is recorded for the decoder
        self.layers_spatial_sizes = [input_maps_len]

        for in_channels, out_channels, key, kernel_size in zip(
            channels[:-1], channels[1:],
            jr.split(key, n_conv_layers),
            kernel_sizes
        ):
            
            # the length of the feature map for the next layer
            conv_output_len = trapallada.util.conv_output_size(conv_output_len, (kernel_size,), (stride,), (padding,))

            # TODO: give a name to the 0-th value that is used below

            self.layers.extend([
                eqx.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, key=key),
                jax.nn.relu,
                eqx.nn.LayerNorm((out_channels, conv_output_len[0]))
            ])

            self.layers_spatial_sizes.append(conv_output_len[0].item())

        # print(f'{out_channels=}, {conv_output_len=}')
        
        self.layers_spatial_sizes.append((out_channels * conv_output_len[0]).item())

        # the last PRNG key from the loop is split again
        _, subkey = jr.split(key)

        self.layers.extend([
            jnp.ravel,
            eqx.nn.Linear(in_features=self.layers_spatial_sizes[-1], out_features=2*dim_z, key=subkey),
            lambda x: x.at[dim_z:].set(jax.nn.softplus(x[dim_z:]) + 1e-3)
            # lambda x: x.at[dim_z:].set(jax.nn.softplus(x[dim_z:]))
        ])
        
    def __call__(self, x: Float[Array, '1 length']) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            x = layer(x)

        return x

    def debug_call(self, x: Float[Array, '1 length']) -> Float[Array, 'twice_dim_z']:

        for layer in self.layers:

            x = layer(x)

            print(f'after {layer.__module__}: {x.shape}')

class Decoder(eqx.Module):

    layers: list

    def __init__(self, input_size: Sequence[int], dim_z: int, layers_spatial_sizes: List[int], kernel_sizes: List[int], channels: List[int], stride: int, key: PRNGKeyArray):

        # for the sake of readability
        n_conv_layers = len(channels)

        assert len(kernel_sizes) == n_conv_layers, f'number of kernel sizes ({len(kernel_sizes)}) must match number of channels ({n_conv_layers})'
        assert len(kernel_sizes) == len(layers_spatial_sizes) - 2,\
            f'number of kernel sizes and channels ({len(kernel_sizes)}) must match that of spatial sizes ({len(layers_spatial_sizes)}) minus 2'

        padding = 0

        # the final number of output channels must be that at the input of the encoder
        # channels = list(reversed(channels)) + [1]
        channels = list(reversed(channels)) + input_size[0:1]

        # the architecture of the encoder is reversed
        kernel_sizes = list(reversed(kernel_sizes))
        layers_spatial_sizes = list(reversed(layers_spatial_sizes))

        self.layers = []

        key, subkey = jr.split(key)

        self.layers.extend([
            eqx.nn.Linear(in_features=dim_z, out_features=layers_spatial_sizes[0], key=subkey),
            jax.nn.relu,
            lambda x: jnp.reshape(x, (channels[0], layers_spatial_sizes[1])), # from a 1D vector to a 2D array
        ])

        # after the `reshape`ing
        conv_output_len = layers_spatial_sizes[1]

        for in_channels, out_channels, key, kernel_size, expected_len in zip(
            channels[:-1], channels[1:],
            jr.split(key, n_conv_layers),
            kernel_sizes,
            layers_spatial_sizes[2:],
        ):
            

            conv_output_len = trapallada.util.transp_conv_output_size(conv_output_len, (kernel_size,), (stride,), (padding,))
            # print(f'{conv_output_len=}, {expected_len=}')

            output_padding = expected_len - conv_output_len[0]
            
            key, subkey = jr.split(key)
            
            self.layers.extend([
                eqx.nn.ConvTranspose1d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    output_padding=output_padding, key=subkey),
                jax.nn.relu,
                eqx.nn.LayerNorm((out_channels, expected_len))
            ])

            conv_output_len += output_padding

        # last `relu` and `LayerNorm` are not needed
        del self.layers[-2:]
        
    def __call__(self, x: Float[Array, 'dim_z']) -> Float[Array, '1 length']:

        for layer in self.layers:

            x = layer(x)

        return x

    def debug_call(self, x: Float[Array, 'dim_z']) -> Float[Array, '1 length']:

        for layer in self.layers:

            x = layer(x)

            print(f'after {layer.__module__}: {x.shape}')

class NonNegativeDecoder(Decoder):

    def __init__(self, input_size: Sequence[int], dim_z: int, layers_spatial_sizes: List[int], kernel_sizes: List[int], channels: List[int], stride: int, key: PRNGKeyArray):

        super().__init__(input_size, dim_z, layers_spatial_sizes, kernel_sizes, channels, stride, key)

        # a RELU layer is added at the end
        self.layers.append(jax.nn.relu)

class ConditionalDecoder(eqx.Module):

    z_decoder: Decoder
    conv_layer: eqx.Module

    def __init__(self, input_size: Sequence[int], dim_z: int, layers_spatial_sizes: List[int], kernel_sizes: List[int], channels: List[int], stride: int, key: PRNGKeyArray):

        key, subkey = jr.split(key)

        assert input_size[0] == 1, 'input size must be (1, length)'

        self.z_decoder = Decoder(input_size, dim_z, layers_spatial_sizes, kernel_sizes, channels, stride, key)
        self.conv_layer = eqx.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, key=subkey)

    # def __init__(self, z_decoder, key: PRNGKeyArray):

    #     key, subkey = jr.split(key)

    #     # assert input_size[0] == 1, 'input size must be (1, length)'

    #     self.z_decoder = z_decoder
    #     self.conv_layer = eqx.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, key=subkey)
        
    #     # a single output channel is expected
    #     assert self.z_decoder.layers[-1].out_channels == 1

    def __call__(self, z: Float[Array, 'dim_z'], x: Float[Array, '1 length']) -> Float[Array, '1 length']:

        return self.z_decoder(z) + self.conv_layer(x)

class VAE(eqx.Module):

    encoder: Encoder
    decoder: Decoder

    def __init__(self, encoder: Encoder, decoder: Decoder):

        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x: Float[Array, 'channel length'], key: PRNGKeyArray) -> PyTree:

        # mean and standard deviation are broken apart...
        z_mean, z_sd = jnp.split(self.encoder(x), 2, axis=-1)

        # ...and used to draw a sample...
        z = dist.Normal(loc=z_mean, scale=z_sd).sample(key)

        # ...which is "decoded" and returned along with the mean and standard deviation
        return self.decoder(z), z_mean, z_sd

def assemble_vae(parameters: dict) -> eqx.Module:

    encoder_class = globals()[parameters['VAE']['encoder']['class']]
    decoder_class = globals()[parameters['VAE']['decoder']['class']]

    encoder = encoder_class(
        **parameters['VAE']['encoder']['parameters'],
        key=jr.PRNGKey(parameters['PRNG seeds']['model initialization']['encoder'])
    )
    
    # `layers_spatial_sizes` is a class attribute filled by the `Encoder`'s `__init__` method
    decoder = decoder_class(
        **(parameters['VAE']['decoder']['parameters'] | {'layers_spatial_sizes': encoder.layers_spatial_sizes}),
        key=jr.PRNGKey(parameters['PRNG seeds']['model initialization']['decoder'])
    )
    
    return VAE(encoder, decoder)

def load_vae(weights_file: str | pathlib.Path, parameters_file: str | pathlib.Path) -> eqx.Module:

    weights_file = pathlib.Path(weights_file)
    assert weights_file.suffix == '.eqx', "weights file should have extension '.eqx'"

    parameters_file = pathlib.Path(parameters_file)

    with parameters_file.open() as f:
        
        parameters = yaml.safe_load(f)
    
    model = assemble_vae(parameters)

    with weights_file.open('rb') as f:
        return eqx.tree_deserialise_leaves(f, model)

def save_vae(file: str | pathlib.Path, model: eqx.Module):
    
    with open(file, 'wb') as f:
        
        eqx.tree_serialise_leaves(f, model)