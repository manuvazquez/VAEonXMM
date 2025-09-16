import pandas as pd

from torch.utils.data import Dataset
from statsmodels.tsa.api import VAR

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

def statsmodels_window_fit(signal: Float[Array, 'channel time'], time_index: pd.DatetimeIndex, p: int = 20, channels_names=[f'RATE_{i}' for i in range(1, 6)]):
    
    # every row a time instant, every column a channel
    # `time_index` is meant to avoid a `UserWarning`
    df = pd.DataFrame(signal.T, columns=channels_names, index=time_index)
    
    # it is differentiated
    # df = df.diff().dropna()

    model = VAR(df,)
    results = model.fit(p,  trend='n')

    # return results.coefs
    return results.coefs.flatten()

def ols_window_fit(signals: Float[Array, 'channels time'], p: int = 2) -> Float[Array, 'coefficient']:
    """
    Fit a VAR model using OLS.
    
    Parameters
    ----------
    signals : Float[Array, 'channels time']
        The input signals.
    p : int
        The order.
        
    Returns
    -------
    Float[Array, 'coefficient']
        The coefficients of the VAR model.
    """

    # so that time runs along the horizontal axis
    signals = signals.T

    # setting up A and b to solve the equation Ax = b
    # i) the desired output
    bs = signals[p:]

    # ii) the input data (each iteration in the loop yields the data for a different lag
    # *for all time instants*, i.e., data points)
    A = jnp.hstack([signals[p - i - 1:len(signals) - i - 1] for i in range(p)])

    x = jnp.linalg.lstsq(A, bs, rcond=None)[0]

    return x.flatten()

def ols_dataset_fit(ds: Dataset, p: int = 2) -> Float[Array, 'coefficient']:
    """
    Fit a VAR model using OLS.
    
    Parameters
    ----------
    ds : Dataset
        The input dataset.
    p : int
        The order.
        
    Returns
    -------
    Float[Array, 'window coefficient']
        The coefficients of the VAR model.
    """

    return jax.vmap(ols_window_fit, in_axes=(0, None))(ds.signals.numpy(), p)