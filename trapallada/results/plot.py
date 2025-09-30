# from typing import Sequence
import pathlib

import pandas as pd
from jaxtyping import Array, Float

import plotly.graph_objects as go
# import plotly.express as px
from plotly.subplots import make_subplots

import trapallada.xsa.util


def full_signals(
    signal: str, path_to_full_signals: str | pathlib.Path,
    window_start: pd.Timestamp | None = None, window_end: pd.Timestamp | None = None,
    predicted_window: Float[Array, 'length'] | None = None):
    """
    Plots specific columns of a DataFrame with red diamonds for False BACKV signals.

    Parameters
    ----------
    signal : str
        The signal to plot.
    path_to_full_signals : str or pathlib.Path
        Path to the full signals data.
    window_start : pandas.Timestamp or None, optional
        Start of the window to highlight.
    window_end : pandas.Timestamp or None, optional
        End of the window to highlight.
    predicted_window : Float[Array, 'length'] or None, optional
        Predicted window data to overlay on the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting plotly figure.
    """

    # either both ends of the window are given or none
    assert ((window_start is None and window_end is None) or (window_start is not None and window_end is not None))

    # signals and their corresponding masks (`DataFrame`s) are loaded
    signals_df, mask_series = trapallada.xsa.util.get_full_signals(signal, path_to_full_signals)

    n_signals = len(trapallada.xsa.util.rates_columns)

    fig = make_subplots(
        rows=n_signals, cols=1,
        shared_xaxes=True, vertical_spacing=0.02, 
        subplot_titles=trapallada.xsa.util.rates_columns)

    for i, (rate_col, backv_col) in enumerate(zip(trapallada.xsa.util.rates_columns, trapallada.xsa.util.background_columns), start=1):
        
        fig.add_trace(
            go.Scatter(x=signals_df.index, y=signals_df[rate_col], mode='lines', name=rate_col),
            row=i, col=1
        )

        false_indices = signals_df.index[~mask_series]
        false_values = signals_df.loc[~mask_series, rate_col]
        fig.add_trace(
            go.Scatter(
                x=false_indices,
                y=false_values,
                mode='markers',
                marker=dict(symbol='diamond', color='red', size=6),
            ),
            row=i, col=1
        )

        if window_start is not None:

            fig.add_vrect(
                x0=window_start, x1=window_end,
                fillcolor="green", opacity=0.2, line_width=0,
                row=i, col=1
            )

    if (window_start is not None) and (predicted_window is not None):
    
        belongs_to_window = (signals_df.index >= window_start) & (signals_df.index <= window_end)
        i_window = signals_df.index[belongs_to_window]

        if belongs_to_window.sum() > predicted_window.shape[1]:

            print('The predicted window is shorter than the actual window (mixing time- and frequency-domain?), prediction won\'t be plotted...')

        else:

            assert len(predicted_window) == n_signals

            for i, pred_rate in enumerate(predicted_window, start=1):
                fig.add_trace(
                    go.Scatter(x=i_window, y=pred_rate, mode='lines', line=dict(dash='dash', width=2, color='gray')),
                    row=i, col=1
                )

    fig.update_layout(
        height=300 * n_signals,
        # title_text="Selected Signals Plot with False BACKV Indicators",
        # xaxis_title="Time",
        showlegend=False
    )
    return fig

def window(signal: str, path_to_full_signals: str | pathlib.Path, window_start: pd.Timestamp, window_end: pd.Timestamp):
    """
    Plots specific columns of a DataFrame with red diamonds for False BACKV signals.

    Parameters
    ----------
    signal : str
        The signal to plot.
    path_to_full_signals : str or pathlib.Path
        Path to the full signals data.
    window_start : pandas.Timestamp
        Start of the window.
    window_end : pandas.Timestamp
        End of the window.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting plotly figure.
    """

    # signals and their corresponding masks (`DataFrame`s) are loaded
    signals_df, mask_series = trapallada.xsa.util.get_full_signals(signal, path_to_full_signals)

    n_signals = len(trapallada.xsa.util.rates_columns)

    fig = make_subplots(
        rows=n_signals, cols=1,
        shared_xaxes=True, vertical_spacing=0.02, 
        subplot_titles=trapallada.xsa.util.rates_columns)

    for i, (rate_col, backv_col) in enumerate(zip(trapallada.xsa.util.rates_columns, trapallada.xsa.util.background_columns), start=1):

        belongs_to_window = (signals_df.index >= window_start) & (signals_df.index <= window_end)
        i_window = signals_df.index[belongs_to_window]
        
        fig.add_trace(
            go.Scatter(x=i_window, y=signals_df.loc[i_window, rate_col], mode='lines', name=rate_col),
            row=i, col=1
        )

        # the region outside the window is shaded gray
        fig.add_vrect(
            x0=signals_df.index[0], x1=window_start,
            fillcolor="gray", opacity=0.2, line_width=0,
            row=i, col=1
        )
        fig.add_vrect(
            x0=window_end, x1=signals_df.index[-1],
            fillcolor="gray", opacity=0.2, line_width=0,
            row=i, col=1
        )

        false_indices = signals_df.index[~mask_series]
        false_values = signals_df.loc[~mask_series, rate_col]
        fig.add_trace(
            go.Scatter(
                x=false_indices,
                y=false_values,
                mode='markers',
                marker=dict(symbol='diamond', color='red', size=6),
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=300 * n_signals,
        # title_text="Selected Signals Plot with False BACKV Indicators",
        # xaxis_title="Time",
        showlegend=False
    )
    return fig

# def roc(
#         labels: Int[Array, 'window'],
#         scores: Float[Array, 'window'],
#         algorithm_name: str | None = None,
#     ):

#     pass