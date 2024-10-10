import pathlib

import yaml

import pandas as pd

from torch.utils.data import Dataset
import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

import plotly.express as px
from IPython.display import display

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

    ds = trapallada.data.SignalsDataset(signals_file, metadata_file, normalize=True)

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

def predict_and_plot(model: eqx.Module, ds: Dataset, i: int, n_samples: int = 2000, rng_key: PRNGKeyArray = jr.PRNGKey(42)):
    
    x = ds[i][1].numpy()
    x_pred, z_mean, z_sd = model(x, rng_key)
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