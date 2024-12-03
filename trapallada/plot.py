from typing import Sequence

import pandas as pd
from jaxtyping import Array, Float
import jax.random as jr

import altair as alt
import plotly.graph_objects as go

def signal(signal: Float[Array, 'length']):

    return alt.Chart(
        pd.DataFrame({'y': signal}).reset_index(names='x')
    ).mark_line().encode(
        x='x', y='y'
    ).properties(width=1_200, height=400)

    # fig = go.Figure()
    
    # fig.add_trace(
    #     go.Scatter(
    #         x=series.index,
    #         y=series,
    #         mode='lines',
    #         name=col
    #     )
    # )


def df(df: pd.DataFrame, columns: Sequence[str] | None = None):
    
    # if no columns names are provided...
    if columns is None:
        
        # ...the original ones are used
        columns = df.columns
    
    else:
        
        assert len(columns) == df.shape[1], 'The number of columns must match that in the DataFrame.'

    fig = go.Figure()

    for col, series in df.set_axis(columns, axis=1).items():

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series,
            mode='lines',
            name=col
        ))

    return fig

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