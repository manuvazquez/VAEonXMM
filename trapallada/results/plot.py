# from typing import Sequence
import pathlib

import pandas as pd
from jaxtyping import Array, Float

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from IPython.display import display, Markdown

import trapallada.xsa.util
# import trapallada.data
import trapallada.plot
import trapallada.results.analysis


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

class TSNE:
    def __init__(self, analysis_df: pd.DataFrame, columns: list[str], full_signals_dir: str | pathlib.Path, on_plot_hover_columns: list[str]):

        self.analysis_df = analysis_df
        self.columns = columns
        self.full_signals_dir = pathlib.Path(full_signals_dir)
        self.on_plot_hover_columns = on_plot_hover_columns

        self.fig = None

    def make_fig(self, color_column: str = 'main_type'):

        self.fig = px.scatter(
            self.analysis_df.reset_index(), # `index` is usually one of the *hover columns*
            *self.columns,
            color=color_column,
            hover_data=self.on_plot_hover_columns, opacity=0.5,
            width=1_200, height=800
        )

        return self.fig
    
    def mark_rectangle(self, x_range: list[float], y_range: list[float], text: str):

        # the rectangle is overimposed on the picture
        self.fig.add_shape(
            type='rect',
            x0=x_range[0], y1=y_range[0], 
            x1=x_range[1], y0=y_range[1],
            line=dict(color='RoyalBlue'),
        )

        self.fig.add_annotation(
            x=x_range[1], y=y_range[1],
            text=text,
            showarrow=False,
            # arrowhead=1
        )

        return self.fig
    
    def sample_and_plot_range(self, n: int, x_range, y_range):

        # `True` if the tSNE coordinates of the corresponding point are inside the rectangle
        belong = (self.analysis_df['tsne_1'] > x_range[0]) &\
            (self.analysis_df['tsne_1'] < x_range[1]) &\
            (self.analysis_df['tsne_2'] < y_range[1]) &\
            (self.analysis_df['tsne_2'] > y_range[0])

        # the original `DataFrame` is subset
        selected_df = self.analysis_df[belong]

        n = min(n, len(selected_df))

        distinct_signals = selected_df['signal'].drop_duplicates()

        for i in distinct_signals.sample(min(n, len(distinct_signals))).index:
            
            display(Markdown(f'### {i}'))
            
            print(self.analysis_df.filter(regex='^(?!z_|tsne_|set$)').loc[i])
            
            signal_id, *start_end = self.analysis_df.loc[i, ['signal', 'start', 'end']]
            display(trapallada.plot.full_signals(signal_id, self.full_signals_dir, *start_end))

    def inspect_regions(self, regions: list[dict[str, tuple[float, float]]], labels, n_samples: int = 5, dont_plot_signals: bool = False):
        
        for region, label in zip(regions, labels):
            
            self.fig = self.mark_rectangle(**region, text=label)

        display(Markdown('## Regions'))
        
        self.fig.show()

        if dont_plot_signals:

            return 

        zoomed_in_fig = go.Figure(self.fig)

        for r, label in zip(regions, labels):

            display(Markdown(f'### {label}'))

            zoomed_in_fig.update_layout(xaxis=dict(range=r['x_range']), yaxis=dict(range=r['y_range']))

            zoomed_in_fig.show()

            self.sample_and_plot_range(n_samples, **r)

    def highlight_window(self, indexes: list[int], name: str):

        self.fig.add_trace(go.Scatter(
            self.analysis_df.loc[indexes, self.columns].
            rename(columns={col: coord for col, coord in zip(self.columns, ['x', 'y'])}).to_dict(orient='list'),
            mode='markers', marker=dict(size=20, symbol='star', color='blue'),
            name=name
        ))

        return self.fig

    def highlight_windows_of_interest(self):
        
        # signals_of_interest = trapallada.data.Labels(
        #     **{k: v for k,v in trapallada.data.labels_files().items() if k == 'reference_signals_file'}
        # ).reference_signals_and_labels
        
        # for signal, name in signals_of_interest:

        #     # in principle, there will more than one
        #     index = self.analysis_df[self.analysis_df['signal'] ==  signal].index

        #     if index.empty:
        #         print(f'{signal} not found')
        #         continue
            
        #     self.highlight_window(index, f'{signal.split("/")[1]} ({name})')
        
        for indexes, name in trapallada.results.analysis.i_windows_of_interest(self.analysis_df):
        
            self.highlight_window(indexes, name)

# def roc(
#         labels: Int[Array, 'window'],
#         scores: Float[Array, 'window'],
#         algorithm_name: str | None = None,
#     ):

#     pass