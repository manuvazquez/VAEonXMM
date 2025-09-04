import pathlib

import yaml
import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Float, Int, Array, PRNGKeyArray

import numpyro.distributions as dist

from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from torch.utils.data import DataLoader, Dataset
import hdbscan

import trapallada.data
import trapallada.model.nn
import trapallada.util

import trapallada.logging as log

logger = log.setup_logging(__name__)

def load_dir(
        dir: str | pathlib.Path, analysis_also: bool=False
    ) -> tuple[trapallada.data.SignalsDataset, pd.DataFrame, eqx.Module, pd.DataFrame, dict] |\
          tuple[trapallada.data.SignalsDataset, pd.DataFrame, eqx.Module, pd.DataFrame, dict, pd.DataFrame]:

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

    # FIXME: for the sake of backwards compatibility (the `normalize` parameter was removed)
    if 'normalize' in parameters['data']:
        del parameters['data']['normalize']

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
def project_onto_latent_space(
        dl: DataLoader, encoder: eqx.Module, return_sd: bool = False
        ) -> Float[Array, 'z'] | tuple[Float[Array, 'index'], Float[Array, 'index']]:
    """
    Build a `DataFrame` with the metadata and the projections of the latent space.

    Args:
        dl (DataLoader): A data loader to be processed.
        encoder (eqx.Module): The encoder.
        return_sd (bool): If True, the standard deviation is also returned.

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

    # # all the `z`s' means are (vertically) stacked together, and the standard deviations dropped
    # z_s = jnp.vstack(z_s)[:, :dim_z]
    
    # all the `z`s' means are (vertically) stacked together
    z_s = jnp.vstack(z_s)

    if return_sd:
        return jnp.split(z_s, 2, axis=-1)
    else:
        return z_s[:, :dim_z]
    
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

def wasserstein_between_one_and_many_gaussians(mu, sd, all_mu, all_sd):
    
    return jnp.sqrt(((mu[None, :] - all_mu)**2).sum(axis=1) + (sd[None, :] + all_sd - 2*jnp.sqrt(sd[None, :] * all_sd)).sum(axis=1))

def wasserstein_between_gaussians(mean: Float[Array, 'i feature'], sd: Float[Array, 'i feature']) -> Float[Array, 'i i']:

    n = mean.shape[0]
    squared_distance = np.zeros((n, n))
    
    for i in range(n):
        
        mean_component = np.sum((mean[i]-mean)**2, axis=1)

        # variance_component = np.sum(sd[i] + sd - 2*np.sqrt(sd[i]*sd), axis=1)
        variance_component = np.clip(np.sum(sd[i] + sd - 2*np.sqrt(sd[i]*sd), axis=1), a_min=0., a_max=None)

        squared_distance[i] = mean_component + variance_component
    
    return np.sqrt(squared_distance)

def wasserstein_cluster(mean: Float[Array, 'i feature'], sd: Float[Array, 'i feature'], n_clusters: int = 4) -> Int[Array, 'i']:

    # compute the Wasserstein distance between the means and standard deviations of the latent space
    d = wasserstein_between_gaussians(mean, sd)

    # compute the hierarchical clustering using the Ward method
    Z = linkage(d[np.triu_indices(len(d),1)], method="ward")

    # assign cluster labels
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    return labels

def data_losses(
        model: eqx.Module, dataloader: DataLoader, x_sd: float, mask_non_gti_intervals: bool, rng_key: PRNGKeyArray):

    log_likelihood, kl_divergence = [], []

    for (_, x, mask), key in zip(dataloader, jr.split(rng_key, len(dataloader))):

        # from torch to numpy/jax
        x = x.numpy()
        mask = mask.numpy()

        batch_size, n_channels, _ = x.shape
        
        overall_times = jnp.array(mask.shape).prod() * n_channels
        overall_good_times = mask.sum() * n_channels

        x_pred, z_mean, z_sd = jax.vmap(model)(x, jr.split(key, batch_size)) # type: ignore

        # if this is a Conditional VAE...
        if isinstance(model, trapallada.model.nn.CVAE):
            
            # ...it is assumed that the last channel is the condition, and hence must not be reconstructed
            x = x.at[:, :-1, :].get()

        assert mask.ndim == 2

        # reconstruction term
        if mask_non_gti_intervals:
            log_likelihood_value = (dist.Normal(loc=x_pred, scale=x_sd).log_prob(x) * mask[:, None, :]).sum(axis=(-1, -2)) / overall_good_times
        else:
            log_likelihood_value = dist.Normal(loc=x_pred, scale=x_sd).log_prob(x).sum(axis=(-1, -2)) / overall_times

        # KL divergence (notice the mask cannot be applied here because it "lives" in the time domain)
        kl_value = -0.5 * jnp.sum(1. + 2.*jnp.log(z_sd) - z_mean**2 - z_sd**2, axis=-1)
    
        # loss_value, (log_likelihood_value, kl_value) = loss_function(model, x, mask, x_sd, beta, mask_non_gti_intervals, subkey)

        log_likelihood.append(log_likelihood_value)
        kl_divergence.append(kl_value)

    return log_likelihood, kl_divergence