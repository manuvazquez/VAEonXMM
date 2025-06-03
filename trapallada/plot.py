from typing import Sequence
import pathlib

import pandas as pd
from jaxtyping import Array, Float

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import trapallada.xsa.util

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

def full_signal(
        signal: str, path_to_full_signals: str | pathlib.Path,
        window_start: pd.Timestamp | None = None, window_end: pd.Timestamp | None = None,
        predicted_window: Float[Array, 'length'] = None
    ):

    # either both ends of the window are given or none
    assert ((window_start is None and window_end is None) or (window_start is not None and window_end is not None))

    signal_series, mask_series = trapallada.xsa.util.get_full_signal(signal, path_to_full_signals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=signal_series.index, y=signal_series, mode='lines', name='signal'))

    try:
        background_series, _ = trapallada.xsa.util.get_full_signal(signal, path_to_full_signals, column='BACKV')
    except KeyError:
        pass
    else:
        fig.add_trace(go.Scatter(
            x=background_series.index, y=background_series, mode='lines', line=dict(dash='dash', width=2, color='gray'), name='background'
        ))

    if window_start is not None:

        belongs_to_window = (signal_series.index >= window_start) & (signal_series.index <= window_end)
        window_series = signal_series[belongs_to_window]
        fig.add_trace(go.Scatter(x=window_series.index, y=window_series, mode='lines', line=dict(color='green', width=2), name='window'))

        if predicted_window is not None:

            if len(window_series) > len(predicted_window):
                print('The predicted window is shorter than the actual window (mixing time- and frequency-domain?), prediction won\'t be plotted...')
            else:
                fig.add_trace(
                    go.Scatter(
                        x=window_series.index, y=predicted_window, mode='lines', line=dict(color='purple', width=2), name='prediction'
                    )
                )

    not_ok_signal = signal_series[~mask_series]
    fig.add_trace(
        go.Scatter(
            x=not_ok_signal.index, y=not_ok_signal, mode='markers', marker=dict(symbol='diamond', color='red', size=2), name='not OK'
        )
    )

    return fig

def pairwise_signals_comparison(
        t: Float[Array, 'length'],
        y1: Float[Array, 'signal length'],
        y2: Float[Array, 'signal length'],
        y1_names: list[str], y2_names: list[str], title: str, xaxis_title: str, yaxis_title: str):
    """
    Compare two signals arrays
    """

    fig = go.Figure()

    for row_y1, row_y2, color, y1_name, y2_name in zip(y1, y2, px.colors.qualitative.Plotly, y1_names, y2_names):
        
        fig.add_trace(go.Scatter(
            x=t,
            y=row_y1,
            # mode='lines+markers',
            mode='lines',
            name=y1_name,
            line=dict(color=color, dash='solid'),
            marker=dict(symbol='circle', color=color)
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=row_y2,
            # mode='lines+markers',
            mode='lines',
            name=y2_name,
            line=dict(color=color, dash='dash'),
            marker=dict(symbol='square', color=color)
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return fig

def full_signals(
    signal: str, path_to_full_signals: str | pathlib.Path,
    window_start: pd.Timestamp | None = None, window_end: pd.Timestamp | None = None):
    """
    Plots specific columns of a DataFrame with red diamonds for False BACKV signals.

    Args:
    signal (str): The signal to plot.
    path_to_full_signals (str | pathlib.Path): Path to the full signals data.
    window_start (pd.Timestamp | None): Start of the window to highlight (optional).
    window_end (pd.Timestamp | None): End of the window to highlight (optional).
    """

    # Load the signals DataFrame and mask series
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

            belongs_to_window = (signals_df.index >= window_start) & (signals_df.index <= window_end)
            # window_series = signals_df[belongs_to_window][rate_col]
            # fig.add_trace(
            #     go.Scatter(x=window_series.index, y=window_series, mode='lines', line=dict(color='green', width=2), name='window'),
            #     row=i, col=1
            #     )
            
            fig.add_vrect(
                x0=window_start, x1=window_end,
                fillcolor="green", opacity=0.2, line_width=0,
                row=i, col=1
            )

    fig.update_layout(
        height=300 * n_signals,
        # title_text="Selected Signals Plot with False BACKV Indicators",
        # xaxis_title="Time",
        showlegend=False
    )
    return fig