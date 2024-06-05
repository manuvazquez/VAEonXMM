from typing import Sequence, Callable, List

import numpy as np
import equinox as eqx

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
