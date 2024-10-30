import pathlib
import re

import yaml

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

import plotly.express as px

from bokeh.models import Scatter
from bokeh.plotting import figure, show
from bokeh.models import GraphRenderer, StaticLayoutProvider, HoverTool, MultiLine, Label

from IPython.display import display, Markdown
import ipywidgets as widgets

import trapallada.data
import trapallada.model.signal
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
    metadata_file = pathlib.Path.home() / parameters['dataset']['files']['metadata_file']

    ds = trapallada.data.SignalsDataset(signals_file, metadata_file, **parameters['data'])

    headers_file = signals_file.with_name('headers.parquet')
    headers_df = pd.read_parquet(headers_file)

    # ================

    weights_file = dir / 'model.eqx'
    assert weights_file.exists()

    model = trapallada.model.signal.load_vae(weights_file, parameters_file)

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

# def display_selected_window(model, ds, index, analysis_df):

#     print(analysis_df.filter(regex='^(?!z_|tsne_|set$)').loc[index])
#     display(predict_and_plot(model, ds, index, display_metadata=False))


# def get_signal_by_id(id: str):
    
#     tar, fit = re_tar_fit.match(id).groups()
    
#     return pd.read_parquet(data_dir / 'full_signals' / (tar + '.parquet'))[fit + '.FTZ']


# def display_selected_signal(model, ds, index, analysis_df):

#     display(trapallada.plot.signal(get_signal_by_id(id).values))


def inspect_regions(fig, n_samples, x_range_list, y_range_list, labels, model, ds, nodes, analysis_df):

    # an auxiliary function to frame a certain region with a rectangle.
    def mark_rectangle(x_range, y_range, fig, text: str):
        
        x_center = np.mean(x_range)
        y_center = np.mean(y_range)
        width = -np.subtract(*x_range)
        height = -np.subtract(*y_range)

        fig.rect(x=x_center, y=y_center, width=width, height=height, line_color='red', line_width=2, fill_alpha=0)

        label = Label(x=x_range[1] + 0.3, y=y_center, text=text, text_font_size="12pt", text_color='black')
        fig.add_layout(label)

        return fig
    

    # another auxiliary function (relying on the previous one) for plotting a few random points in a region
    def sample_and_plot_range(n: int, x_range, y_range):

        belong = (x_range[0]< nodes['x']) & (nodes['x'] < x_range[1]) &\
            (y_range[0]< nodes['y']) & (nodes['y'] < y_range[1])
        
        selected = nodes[belong]

        n = min(n, len(selected))

        for i in selected.sample(n, random_state=42).index:
            display(Markdown(f'### {i}'))
            print(analysis_df.filter(regex='^(?!z_|tsne_|set$)').loc[i])
            display(trapallada.analysis.predict_and_plot(model, ds, i, display_metadata=False))
    
    
    for x_range, y_range, label in zip(x_range_list, y_range_list, labels):
        
        fig = mark_rectangle(x_range, y_range, fig, label)

    show(fig)

    for x_range, y_range, label in zip(x_range_list, y_range_list, labels):

        display(Markdown(f'### {label}'))
        
        fig.x_range.start, fig.x_range.end = x_range
        fig.y_range.start, fig.y_range.end = y_range
        
        show(fig)

        sample_and_plot_range(n_samples, x_range, y_range)

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