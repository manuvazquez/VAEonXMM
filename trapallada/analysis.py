import pathlib
import re

import yaml

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array, PRNGKeyArray

import plotly.express as px
import plotly.graph_objects as go

import altair as alt

from bokeh.models import Scatter, Circle, IndexFilter, CDSView, LabelSet, ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.models import GraphRenderer, StaticLayoutProvider, HoverTool, MultiLine, Label

from IPython.display import display, Markdown
import ipywidgets as widgets

import trapallada.data
import trapallada.model.nn
import trapallada.plot

def load_results_dir(dir, analysis_also: bool=False):

    # it's turned into a `Pathlib` object
    dir = pathlib.Path(dir)

    # ================

    parameters_file = dir / 'extended_parameters.yaml'
    assert parameters_file.exists()

    with parameters_file.open() as f:
        parameters = yaml.safe_load(f)

    # ---

    signals_file = pathlib.Path.home() / parameters['dataset']['files']['signals_file']
    # FIXME: this is a temporary fix
    # masks_file = pathlib.Path.home() / parameters['dataset']['files']['masks_file']
    masks_file = signals_file.with_stem('masks')
    metadata_file = pathlib.Path.home() / parameters['dataset']['files']['metadata_file']

    ds = trapallada.data.SignalsDataset(signals_file, metadata_file, masks_file, **parameters['data'])

    headers_file = signals_file.with_name('headers.parquet')
    headers_df = pd.read_parquet(headers_file)

    # ================

    weights_file = dir / 'model.eqx'
    assert weights_file.exists()

    model = trapallada.model.nn.load_vae(weights_file, parameters_file)

    # ================
    
    results_file = dir / 'in_training_metrics.parquet'
    assert results_file.exists()

    results_df = pd.read_parquet(results_file)

    # ================

    to_be_returned = ds, headers_df, model, results_df, parameters

    # ================

    if analysis_also:

        analysis_file = dir / 'analysis.parquet'
        assert analysis_file.exists()
        analysis_df = pd.read_parquet(analysis_file)

        to_be_returned += (analysis_df,)


    return to_be_returned

# =============================================================

def predict_and_plot(
        model: eqx.Module, ds: Dataset,
        i: int, n_samples: int = 2000,
        display_metadata: bool = True,
        rng_key: PRNGKeyArray = jr.PRNGKey(42)
    ):
    
    x = ds[i][1].numpy()
    x_pred, z_mean, z_sd = model(x, rng_key)
    if display_metadata:
        display(ds.metadata_df.loc[i])
    df = pd.DataFrame({'prediction': x_pred[0], 'actual': x.ravel()})
    
    return trapallada.plot.df(df.iloc[:n_samples])

def plot_signal(
        model: eqx.Module, ds: Dataset,
        analysis_df: pd.DataFrame, signal_file: str,
        hover_data: list[str], x_range: tuple[float], y_range: tuple[float]
    ):
    
    df = analysis_df[analysis_df['signal'] == signal_file]

    fig = px.scatter(df.reset_index(), x='tsne_1', y='tsne_2', text='# window', hover_data=hover_data, width=1_200, height=800)
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis=dict(range=x_range), yaxis=dict(range=y_range))
    fig.show()

    for i, row in df.iterrows():
        print(f'# window: {row['# window']}',)
        display(predict_and_plot(model, ds, i))

class Graph:

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, fig=None):

        self.nodes_df = nodes
        self.edges_df = edges

        self.fig = fig

        self.re_columns_to_show = r'^(?!z_|tsne_|set$)'

    def make_fig(self, edges_line_alpha: float = 0.01, fig=None):

        if self.fig is None:

            self.fig = figure(title="Latent space", width=1_200, height=1024)

        graph = GraphRenderer()

        graph.node_renderer.glyph = Scatter(
            size='size', fill_color='fill_color', marker='marker',
            line_alpha='alpha', fill_alpha='alpha',
        )

        graph.node_renderer.data_source.data = self.nodes_df.reset_index()[[
            'index', 'fill_color', 'marker', 'size',
            'alpha',
            ]].to_dict(orient='list')

        graph.node_renderer.data_source.data['community'] = self.nodes_df['community'].to_list()

        tooltips=[
            ("Community", "@community"),
            ("index", "@index"),
        ]

        if 'main_type' in self.nodes_df:
            
            graph.node_renderer.data_source.data['main_type'] = self.nodes_df['main_type'].to_list()
            
            tooltips.append(("main_type", "@main_type"))
        
        hover_tool = HoverTool(tooltips=tooltips)

        graph.edge_renderer.data_source.data = self.edges_df.to_dict('list')
        graph.edge_renderer.glyph = MultiLine(line_color="black", line_alpha=edges_line_alpha)

        graph.layout_provider = StaticLayoutProvider(graph_layout=self.nodes_df[['x', 'y']].apply(lambda row: row.tolist(), axis=1).to_dict())

        self.fig.renderers.append(graph)
        self.fig.add_tools(hover_tool)

        return self.fig
    
    def mark_rectangle(self, x_range, y_range, text: str):
    
        x_center = np.mean(x_range)
        y_center = np.mean(y_range)
        width = -np.subtract(*x_range)
        height = -np.subtract(*y_range)

        self.fig.rect(x=x_center, y=y_center, width=width, height=height, line_color='red', line_width=2, fill_alpha=0)

        label = Label(x=x_range[1] + 0.3, y=y_center, text=text, text_font_size="12pt", text_color='black')
        self.fig.add_layout(label)

        return self.fig
    
    def sample_and_plot_range(self, n_samples: int, x_range, y_range, model: eqx.Module, ds: Dataset):

        belong = (x_range[0]< self.nodes_df['x']) & (self.nodes_df['x'] < x_range[1]) &\
            (y_range[0]< self.nodes_df['y']) & (self.nodes_df['y'] < y_range[1])
        
        selected = self.nodes_df[belong]

        n_samples = min(n_samples, len(selected))

        for i in selected.sample(n_samples, random_state=42).index:
            
            display(Markdown(f'### {i}'))
            
            # print(self.nodes_df.filter(regex=self.re_columns_to_show).loc[i])
            
            # display(trapallada.analysis.predict_and_plot(model, ds, i, display_metadata=False))

            self.display_selected(model, ds, i)

    def display_selected(self, model: eqx.Module, ds: Dataset, i: int):

        print(self.nodes_df.filter(regex=self.re_columns_to_show).loc[i])
        
        display(predict_and_plot(model, ds, i, display_metadata=False))

    def inspect_regions(self, n_samples, regions: list[dict[str, tuple[float, float]]], labels, model: eqx.Module, ds: Dataset):

        for r, l in zip(regions, labels):

            fig = self.mark_rectangle(**r, text=l)

        show(fig)

        for r, l in zip(regions, labels):

            display(Markdown(f'### {l}'))
            
            fig.x_range.start, fig.x_range.end = r['x_range']
            fig.y_range.start, fig.y_range.end = r['y_range']
            
            show(fig)

            self.sample_and_plot_range(n_samples, **r, model=model, ds=ds)

    def highlight_nodes(self, nodes_labels: list[tuple[str, str]]):

        highlight_renderer = GraphRenderer()

        highlighted_nodes, highlighted_nodes_labels = list(zip(*nodes_labels))

        # data already in the graph
        data = dict(self.fig.renderers[0].node_renderer.data_source.data)

        highlight_renderer.node_renderer.data_source.data = data
        highlight_renderer.layout_provider = self.fig.renderers[0].layout_provider
        highlight_renderer.node_renderer.glyph = Circle(radius=20, fill_color="yellow", line_color="black", line_width=3)

        # so that only the highlighted nodes are shown
        highlighted_indices = [data['index'].index(node) for node in highlighted_nodes if node in data['index']]
        highlight_filter = IndexFilter(indices=highlighted_indices)
        highlight_view = CDSView(filter=highlight_filter)
        highlight_renderer.node_renderer.view = highlight_view

        self.fig.renderers.append(highlight_renderer)

        # data for labels
        highlighted_positions = {'x': [], 'y': [], 'node': []}
        for node, label in zip(highlighted_nodes, highlighted_nodes_labels):
            if node in data['index']:
                # index = data['index'].index(node)
                x, y = highlight_renderer.layout_provider.graph_layout[node]
                highlighted_positions['x'].append(x)
                highlighted_positions['y'].append(y)
                highlighted_positions['node'].append(label)

        # Create a ColumnDataSource for the labels
        label_source = ColumnDataSource(data=highlighted_positions)

        # Add the labels using LabelSet
        labels = LabelSet(x='x', y='y', text='node', source=label_source, text_font_size='10pt',
                        text_color='black', background_fill_color='white', background_fill_alpha=0.7)

        # Add labels to the figure
        self.fig.add_layout(labels)

        show(self.fig)


def plot_graph(nodes: pd.DataFrame, edges: pd.DataFrame, analysis: pd.DataFrame | None = None, edges_line_alpha: float = 0.01, fig=None):

    if fig is None:

        fig = figure(title="Latent space", width=1_200, height=1024)

    graph = GraphRenderer()

    # graph.node_renderer.glyph = Scatter(size='size', fill_color='fill_color', marker='marker')
    graph.node_renderer.glyph = Scatter(
        size='size', fill_color='fill_color', marker='marker',
        line_alpha='alpha', fill_alpha='alpha',
    )

    graph.node_renderer.data_source.data = nodes.reset_index()[[
        'index', 'fill_color', 'marker', 'size',
        'alpha',
        ]].to_dict(orient='list')

    graph.node_renderer.data_source.data['community'] = nodes['community'].to_list()

    tooltips=[
        ("Community", "@community"),
        ("index", "@index"),
    ]

    # if an *analysis* `DataFrame` was passed...
    if analysis is not None:
        
        graph.node_renderer.data_source.data['main_type'] = analysis['main_type'].to_list()
        
        tooltips.append(("main_type", "@main_type"))
    
    hover_tool = HoverTool(tooltips=tooltips)

    graph.edge_renderer.data_source.data = edges.to_dict('list')
    graph.edge_renderer.glyph = MultiLine(line_color="black", line_alpha=edges_line_alpha)

    graph.layout_provider = StaticLayoutProvider(graph_layout=nodes[['x', 'y']].apply(lambda row: row.tolist(), axis=1).to_dict())

    fig.renderers.append(graph)
    fig.add_tools(hover_tool)

    return fig

class SignalsGraph(Graph):

    # a regular expression to extract the names of the `.tar` file and inner `.FTZ` from some `Id`
    re_tar_fit = re.compile(r'(\w+)/(\w+)')

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, data_dir: str | pathlib.Path, fig=None):

        super().__init__(nodes, edges, fig)

        self.data_dir = pathlib.Path(data_dir)

    def get_signal_by_id(self, id: str):
        
        tar, fit = self.re_tar_fit.match(id).groups()
        
        return pd.read_parquet(self.data_dir / 'full_signals' / (tar + '.parquet'))[fit + '.FTZ']

    def display_selected(self, model: eqx.Module, ds: Dataset, i: int):

        display(trapallada.plot.signal(self.get_signal_by_id(i).values))

def make_combo_widget(results_dir: pathlib.Path):
    
    if results_dir.exists():
    
        subdirs = [e.relative_to(results_dir).as_posix() for e in results_dir.iterdir() if e.is_dir()]

    else:
        
        subdirs = []
    
    combo = widgets.Combobox(
        placeholder='Type or select an option',
        options=subdirs,
        description='Choose data directory:',
        ensure_option=False,
        # disabled=False
    )
    
    combo.style.description_width = 'initial'
    
    return combo

def get_base_directory():
    
    base_dir = pathlib.Path.cwd()
    
    while not (base_dir / '.git').exists():
        
        base_dir = base_dir.parent

    return base_dir

def save_regions(regions: list[dict], output_file: pathlib.Path | str) -> pathlib.Path:

    with open(output_file, 'w') as f:
        
        yaml.dump(regions, f, default_flow_style=False)

def load_regions(regions_file: pathlib.Path) -> list[dict]:
    
    with regions_file.open() as f:
        
        return yaml.load(f, Loader=yaml.FullLoader)

class TSNE:
    def __init__(self, analysis_df: pd.DataFrame, columns: list[str], full_signals_dir: str | pathlib.Path, on_plot_hover_columns: list[str]):

        self.analysis_df = analysis_df
        self.columns = columns
        self.full_signals_dir = pathlib.Path(full_signals_dir)
        self.on_plot_hover_columns = on_plot_hover_columns

        self.fig = None

    def make_fig(self, color_column: str = 'main_type'):

        self.fig = px.scatter(
            self.analysis_df.reset_index(),
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

        for i in selected_df.sample(n, random_state=42).index:
            
            display(Markdown(f'### {i}'))
            
            print(self.analysis_df.filter(regex='^(?!z_|tsne_|set$)').loc[i])
            
            # display(predict_and_plot(model, ds, i, display_metadata=False))
            # display(plot_window_within_signal(i, self.full_signals_dir, self.analysis_df))
            signal_id, *start_end = self.analysis_df.loc[i, ['signal', 'start', 'end']]
            display(plot_full_signal(signal_id, self.full_signals_dir, *start_end))

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

    def highlight_windows(self, indexes: list[int], name: str):

        self.fig.add_trace(go.Scatter(
            self.analysis_df.loc[indexes, self.columns].
            rename(columns={col: coord for col, coord in zip(self.columns, ['x', 'y'])}).to_dict(orient='list'),
            mode='markers', marker=dict(size=20, symbol='star', color='blue'),
            # customdata=self.analysis_df.loc[indexes, self.on_plot_hover_columns],
            name=name
        ))

        return self.fig

    def highlight_windows_of_interest(self):
        
        signals_of_interest = get_signals_of_interest()
        
        for signal, name in signals_of_interest:
            
            # self.highlight_windows(self.analysis_df[self.analysis_df['signal'] ==  signal].index, name)
            self.highlight_windows(self.analysis_df[self.analysis_df['signal'] ==  signal].index, f'{signal.split("/")[1]} ({name})')

def get_signals_of_interest():
    
    with (get_base_directory() / 'interesting.yaml').open() as f:
        
        return yaml.safe_load(f)['signals']

def get_full_signal(signal: str, path_to_full_signals: str | pathlib.Path) -> tuple[pd.Series, pd.Series]:
    
    tar_file, fit_file = signal.split('/')

    path_to_full_signals = pathlib.Path(path_to_full_signals)

    signal_series = pd.read_parquet(path_to_full_signals / f'{tar_file}.parquet')[(f'{fit_file}.FTZ', 'RATE')]
    mask_series = pd.read_parquet(path_to_full_signals / 'mask' / f'{tar_file}.parquet')[(f'{fit_file}.FTZ', 'OK')]

    return signal_series, mask_series

def plot_full_signal(
        signal: str, path_to_full_signals: str | pathlib.Path,
        window_start: pd.Timestamp | None = None, window_end: pd.Timestamp | None = None,
        predicted_window: Float[Array, 'length'] = None
    ):

    # either both ends of the window are given or none
    assert ((window_start is None and window_end is None) or (window_start is not None and window_end is not None))

    signal_series, mask_series = get_full_signal(signal, path_to_full_signals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=signal_series.index, y=signal_series, mode='lines', name='signal'))

    if window_start is not None:

        belongs_to_window = (signal_series.index >= window_start) & (signal_series.index <= window_end)
        window_series = signal_series[belongs_to_window]
        fig.add_trace(go.Scatter(x=window_series.index, y=window_series, mode='lines', line=dict(color='green', width=2), name='window'))

        if predicted_window is not None:

            # assert len(predicted_window) == len(window_series)
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

class Latents:

    def __init__(self, dl, encoder: eqx.Module, dim_z: int):

        self.dim_z = dim_z
        
        z_s = []
        
        for _, x, _ in dl:
            z_s.append(jax.vmap(encoder)(x.numpy()))

        self.z_s = jnp.vstack(z_s)

    def plot_ith(self, i_z: int):

        fig = px.scatter(x=jnp.arange(self.dim_z), y=self.z_s[i_z, :self.dim_z], error_y=self.z_s[i_z, self.dim_z:])
    
        return fig

def make_save_button_widget(regions, regions_file):
    
    # Define the function to be called when the button is pressed
    def on_button_click(b):
        save_regions(regions, regions_file)
        print("Regions saved to", regions_file)

    # Create the button
    button = widgets.Button(description="Save Regions")

    # Set the button's on_click event to the function
    button.on_click(on_button_click)

    # Display the button
    display(button)

def check_window(i: int, model: eqx.Module, ds: Dataset, analysis_df: pd.DataFrame, full_signals_dir: str | pathlib.Path) -> go.Figure:
    
    x = ds[i][1].numpy()
    
    x_pred, z_mean, z_sd = model(x, jr.PRNGKey(42))
    
    signal_id, *start_end = analysis_df.loc[i, ['signal', 'start', 'end']]
    
    return plot_full_signal(signal_id, full_signals_dir, *start_end, x_pred.squeeze())

def metric_chart(results_df: pd.DataFrame, metric: str, n_rows: int = 4_000):

    sub_df = results_df.loc[:n_rows,[('training', metric), ('validation', metric)]]
    sub_df.columns = sub_df.columns.get_level_values(0)
    chart = alt.Chart(
        sub_df.reset_index()
    ).transform_fold(
        sub_df.columns.to_list(), ['dataset', metric]
    ).mark_line().encode(
        x='index:Q', y=f'{metric}:Q', color='dataset:N'
    ).properties(width=600, height=400)

    return chart


def plot_samples_from_cluster(analysis_df: pd.DataFrame, full_signals_dir: str | pathlib.Path, cluster_column: str = 'cluster in z', n: int = 5):
    
    for i_cluster in analysis_df['cluster in z'].value_counts()[1:].index:

        display(Markdown(f'## Cluster {i_cluster}'))

        selected_df = analysis_df[analysis_df[cluster_column] == i_cluster]
        
        for i in selected_df.sample(n, random_state=42).index:
            
            display(Markdown(f'### {i}'))
            
            print(analysis_df.filter(regex='^(?!z_|tsne_|set$)').loc[i])
            
            signal_id, *start_end = analysis_df.loc[i, ['signal', 'start', 'end']]
            display(trapallada.analysis.plot_full_signal(signal_id, full_signals_dir, *start_end))
