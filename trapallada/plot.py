from typing import Sequence

import pandas as pd
from jaxtyping import Array, Float
import jax.random as jr

import altair as alt

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