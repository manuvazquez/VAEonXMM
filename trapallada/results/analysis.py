import pathlib

import pandas as pd
import equinox as eqx

from torch.utils.data import DataLoader, Dataset
import hdbscan

import trapallada.util


def cluster(ds: Dataset, encoder: eqx.Module) -> pd.DataFrame:

    # a `DataLoader` for the dataset...
    dl = DataLoader(ds, batch_size=128, shuffle=False)

    # ...the latter being encoded into the latent space (last columns in the resulting `DataFrame`)
    res_df = trapallada.util.make_projections_df(ds, dl, encoder)

    def columns_clusters(selected_cols: pd.DataFrame, min_cluster_size: int):

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        
        labels = clusterer.fit_predict(selected_cols)
        
        return pd.Categorical(labels)

    # clustering in both:

    # i) the latent space...
    res_df['cluster in z'] = columns_clusters(res_df.filter(like='z_'), 10)
    
    # ii) ... the projected space (t-SNE)
    res_df['cluster in projected space'] = columns_clusters(res_df.filter(like='tsne_'), 80)

    return res_df

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