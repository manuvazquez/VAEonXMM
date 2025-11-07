# from typing import Sequence
import pathlib

import pandas as pd
from jaxtyping import Array, Float, Int
import equinox as eqx

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import roc_curve

from IPython.display import display, Markdown

import trapallada.xsa.util
import trapallada.data
import trapallada.plot
import trapallada.results.analysis

import marimo as mo


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

    def make_fig(self, color_column: str | None = 'main_type'):

        self.fig = px.scatter(
            self.analysis_df.reset_index(), # `index` is usually one of the *hover columns*
            *self.columns,
            color=color_column,
            hover_data=self.on_plot_hover_columns, opacity=0.5,
            width=1_200, height=800
        )

        return self.fig
    
    def mark_rectangle(self, x_range: list[float], y_range: list[float], text: str):

        assert self.fig is not None, 'Figure not created yet. Call `make_fig()` first.'

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
            display(full_signals(signal_id, self.full_signals_dir, *start_end))

    def inspect_regions(
            self,
            regions: list[dict[str, tuple[float, float]]],
            labels, n_samples: int = 5, dont_plot_signals: bool = False
        ):

        assert self.fig is not None, 'Figure not created yet. Call `make_fig()` first.'
        
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

    def highlight_window(self, indexes: list[int] | pd.Index, name: str):

        assert self.fig is not None, 'Figure not created yet. Call `make_fig()` first.'

        self.fig.add_trace(go.Scatter(
            self.analysis_df.loc[indexes, self.columns].
            rename(columns={col: coord for col, coord in zip(self.columns, ['x', 'y'])}).to_dict(orient='list'),
            mode='markers', marker=dict(size=20, symbol='star', color='blue'),
            name=name
        ))

        return self.fig

    def highlight_windows_of_interest(self):
        
        for indexes, name in trapallada.results.analysis.i_windows_of_interest(self.analysis_df):
        
            self.highlight_window(indexes, name)

def marimo_selected(
        mo_plot,
        analysis_df: pd.DataFrame, 
        model: eqx.Module,
        model_name: str,
        full_signals_dir: str | pathlib.Path,
        max_n_samples_to_show: int = 30,
        re_drop_columns: str = '^(?!z_|tsne_|set$|RA|DEC|CRA|CDEC|signal)'
    ) -> pd.DataFrame:

    i_selected = [e['index'] for e in mo_plot.points if e]

    plotted_samples = analysis_df.loc[i_selected]

    # belongs_to_reference_signals = plotted_samples['signal'].isin(trapallada.data.Labels().signals)
    belongs_to_reference_signals = plotted_samples['signal'].isin(
        trapallada.data.Labels(**trapallada.data.labels_files()).signals
    )
    print(f'dismissed {belongs_to_reference_signals.sum()} window(s) belonging to reference or annotated signals...')
    plotted_samples = plotted_samples.drop(plotted_samples[belongs_to_reference_signals].index)

    if len(plotted_samples) > max_n_samples_to_show:
        plotted_samples = plotted_samples.drop_duplicates(subset=['signal'])
        print('Too many selected, plotting just one window for every signal...')
        if len(plotted_samples) > max_n_samples_to_show:
            plotted_samples = plotted_samples.sample(max_n_samples_to_show)
            print(f'Still too many, plotting {max_n_samples_to_show} randomly selected...')

    # output_tables = []
    
    for i_row, row in plotted_samples.iterrows():
        mo.output.append(
            row.filter(regex=re_drop_columns).to_frame().T
        )
        # output_tables.append(mo.ui.table(
        #     row.filter(regex=re_drop_columns).to_frame().T
        # ))
        # mo.output.append(output_tables[-1])
        signal_id, *start_end = row[['signal', 'start', 'end']]
        if model_name == 'vae':
            pred = model.decoder(plotted_samples.filter(regex=r'z_\d+').loc[i_row])
        else:
            pred = None
        full_signal_plot_fig = full_signals(signal_id, full_signals_dir, *start_end, pred)
        mo.output.append(full_signal_plot_fig)
        mo.output.append('---')
    
    return plotted_samples
    # return plotted_samples, output_tables


def roc(
        labels: Int[Array, 'window'],
        scores: Float[Array, 'window'],
        title: str = 'ROC Curve',
        return_plot_data: bool = False
    ):

    fpr, tpr, thresholds = roc_curve(labels, scores)

    fig = px.area(
        x=fpr, y=tpr,
        title=title,
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=500, height=500,
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    if return_plot_data:

        df = pd.DataFrame({
            'FPR': fpr,
            'TPR': tpr,
            'Thresholds': thresholds
        })

        return fig, df
    
    else:
        
        return fig