import pathlib

import yaml
import pandas as pd

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
import hdbscan

import trapallada.data
import trapallada.model.nn
import trapallada.util

import trapallada.logging as log

logger = log.setup_logging(__name__)

def load_dir(dir, analysis_also: bool=False):

    # it's turned into a `Pathlib` object
    dir = pathlib.Path(dir)

    # ================

    parameters_file = dir / 'extended_parameters.yaml'
    assert parameters_file.exists()

    with parameters_file.open() as f:
        parameters = yaml.safe_load(f)

    # ---

    signals_file = pathlib.Path.home() / parameters['dataset']['files']['signals_file']
    
    # FIXME: a temporary fix for compatibility with the old code
    masks_file = signals_file.with_stem('masks')
    # masks_file = pathlib.Path.home() / parameters['dataset']['files']['masks_file']
    
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


def hdbscan_cluster(selected_cols: pd.DataFrame, min_cluster_size: int):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    
    labels = clusterer.fit_predict(selected_cols)
    
    return pd.Categorical(labels)

def augment_with_positions(df: pd.DataFrame, positions_df: pd.DataFrame, pos_columns: list[str] = ['RA', 'DEC', 'CRA', 'CDEC'], tsne_columns: list[str] = ['tsne_1', 'tsne_2']) -> pd.DataFrame:

    # input `DataFrame` is extended with positions; notice the indexes in `df` are kept
    res = df.join(positions_df, on=['tar file', 'source'], how='left')
    
    assert not res.isna().any().any()

    # in order to share the result with a collaborator, who will use it to query database;
    # notice that, actually, `positions_df` would be enough
    # res[pos_columns + tsne_columns].to_csv('pos_and_proj.csv', index=False)

    return res

def augment_with_simbad(df: pd.DataFrame, catalog_file: str | pathlib.Path, columns_of_interest: list[str] = ['main_type', 'other_types', 'main_id'], pos_columns: list[str] = ['RA', 'DEC', 'CRA', 'CDEC']) -> pd.DataFrame:

    # an external catalog with some *alleged* labels associated with spatial coordinates
    catalog_file = pathlib.Path(catalog_file)
    assert catalog_file.exists()

    catalog_df = pd.read_csv(catalog_file)

    # there might be a number of rows (labels) for every position in the catalog (different things can be 
    # identified *at different time instants*); only the first is kept
    unique_pos_catalog_df = catalog_df.groupby(pos_columns).first().reset_index()

    res = pd.merge(
        df, unique_pos_catalog_df[columns_of_interest + pos_columns],
        on=pos_columns, how='left'
    )

    return res

def augment_with_window_and_signal_ids(df: pd.DataFrame) -> pd.DataFrame:

    res = df.copy()

    # the number of window *within* every (*tar* file, *FIT* file) combination
    res['# window'] = res.groupby(['tar file', 'source']).cumcount()

    # the *tar* and *FIT* file, which univocally determines the signal, are combined into a single column
    res['signal'] = res['tar file'].str.replace('.tar', '') + '/' + res['source'].str.replace('.FTZ', '')

    for group_name, group_df in res.groupby('signal'):
        assert len(group_df['main_type'].unique()) == 1

    return res

# def project_onto_latent_space(dl: DataLoader, encoder: eqx.Module) -> tuple[Float[Array, 'z'], Float[Int, 'index']]:
def project_onto_latent_space(dl: DataLoader, encoder: eqx.Module) -> Float[Array, 'z']:
    """
    Build a `DataFrame` with the metadata and the projections of the latent space.

    Args:
        dl (DataLoader): A data loader to be processed.
        encoder (eqx.Module): The encoder.

    Returns:
        - z_s: A JAX array representing the stacked latent space projections.
    """
    
    # the dataset associated with `dl` is encoded, and the corresponding indexes (in the full datset) recorded
    # z_s, i_s = [], []
    z_s = []
    for i, x, *_ in dl:
        z_s.append(jax.vmap(encoder)(x.numpy()))
        # i_s.append(i.numpy())

    # for the sake of clarity (assuming `z` comprises mean and standard deviation)
    dim_z = z_s[-1].shape[-1] // 2

    # all the `z`s' means are (vertically) stacked together, and the standard deviations dropped
    z_s = jnp.vstack(z_s)[:, :dim_z]

    return z_s
    # return z_s, jnp.hstack(i_s)

def tsne_project(z_s: Float[Array, 'window z'], n_projected_components: int = 2, random_state: int = 42) -> Float[Array, 'window projection']:
    """
    Build a `DataFrame` with the metadata and the projections of the latent space.

    Args:
        n_projected_components (int): The number of components for the t-SNE embeddings.

    Returns:
        A `DataFrame` with the metadata and the projections of the latent space.
    """

    dim_z = z_s.shape[-1]

    if dim_z > 2:

        # t-SNE is applied to the (means of the) `z`s
        t_sne_embeddings = jnp.array(TSNE(
            n_components=n_projected_components, perplexity=30, random_state=random_state).fit_transform(z_s)
        )
    
    elif dim_z == 2:

        t_sne_embeddings = z_s
        logger.warning('The latent space has only two dimensions: setting t-SNE projections equal to it...')

    else:

        raise ValueError('The latent space must have at least two dimensions.')


    return t_sne_embeddings