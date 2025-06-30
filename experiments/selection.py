import marimo

__generated_with = "0.14.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> Check the signals in some clusters""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pathlib

    import numpy as np
    import pandas as pd
    import scipy.spatial.distance

    from torch.utils.data import DataLoader

    import plotly
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go

    import trapallada.analysis
    import trapallada.results.analysis
    import trapallada.results.plot
    import trapallada.model.ar
    return np, pathlib, pd, scipy, trapallada


@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = 'notebook'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Parameters""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Some global parameters""")
    return


@app.cell
def _():
    n_signals_to_keep_from_every_model = 15

    n_random_signals_to_keep = 80
    return n_random_signals_to_keep, n_signals_to_keep_from_every_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model path""")
    return


@app.cell
def _(mo, pathlib):
    file_browser = mo.ui.file_browser(
        initial_path=pathlib.Path('results/'), selection_mode='directory', multiple=False, restrict_navigation=True
    )
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, mo):
    model_dir = file_browser.path()
    mo.stop(model_dir is None, mo.md("**Choose a Model Path.**"))
    return (model_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Analysis data* produced here can be associated with a particular trained model (for the case of VAEs) or with the `Dataset` (for the case of the AR model). The name of the file is also dependent on the model.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Input Files""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The base directory""")
    return


@app.cell
def _():
    # base_dir = trapallada.util.get_base_directory()
    # base_dir
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Particular *sub*directory with the trained model. It should be a `str` so that parameters can be injected with papermill.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    From the results directory,

    - the `Dataset`,
    - the headers of the corresponding files,
    - the ML model,
    - a `DataFrame` with results from training, 
    - a `dict` with the parameters

    are obtained
    """
    )
    return


@app.cell
def _(model_dir, trapallada):
    ds, positions_df, model, training_df, parameters = trapallada.results.analysis.load_dir(model_dir)
    return ds, parameters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""From the parameters of the simulation it's possible to find out the directory containing the *full* signals (before splitting into *windows*).""")
    return


@app.cell
def _(parameters, pathlib):
    full_signals_dir = (pathlib.Path.home() / parameters['dataset']['files']['signals_file']).parent / 'full_signals'
    full_signals_dir
    return (full_signals_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Metadata is extracted from the `Dataset`. A new column `signal` is added for the sake of convenience.""")
    return


@app.cell
def _(ds):
    metadata_df = ds.metadata_df.copy()
    metadata_df['signal'] = metadata_df['tar file'].str.replace('.tar', '') + '/' + metadata_df['source'].str.replace('.FTZ', '')
    return (metadata_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A single window for each (selected beforehand) *signal of interest*""")
    return


@app.cell
def _(metadata_df, trapallada):
    signals_of_interest_df = metadata_df[
        metadata_df['signal'].isin(trapallada.data.Labels().signals)
    ]
    signals_of_interest_df.drop_duplicates('source')
    return (signals_of_interest_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*All the windows*, to be selected for plotting""")
    return


@app.cell
def _(mo, signals_of_interest_df):
    # signals_of_interest_table = mo.ui.table(signals_of_interest_df, selection='single')
    signals_of_interest_table = mo.ui.table(signals_of_interest_df)
    signals_of_interest_table
    return (signals_of_interest_table,)


@app.cell
def _(full_signals_dir, mo, signals_of_interest_table, trapallada):
    # mo.ui.dataframe(signals_of_interest_df)
    for i_row, row in signals_of_interest_table.value.iterrows():
        signal_id, *start_end = row[['signal', 'start', 'end']]
        mo.output.append(trapallada.results.plot.full_signals(signal_id, full_signals_dir, *start_end))
        mo.output.append('---')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A button that, when pressed, will save the current selection in an *Excel* file.""")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save")
    save_button
    return (save_button,)


@app.cell
def _(save_button, signals_of_interest_table):
    selection_saved_file = 'selected.xlsx'
    if save_button.value:
        signals_of_interest_table.value.to_excel(selection_saved_file, index_label='# window')
    return (selection_saved_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The selection that is currently saved. **CAVEAT**: `# window` here is not the number of window *within* a signal but the overall number of window (more of an identifier).""")
    return


@app.cell
def _(pd, selection_saved_file):
    selected_df = pd.read_excel(selection_saved_file)
    selected_df.head()
    return (selected_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A function to get the closest windows to the selected ones, as given by the passed *analysis* file""")
    return


@app.cell
def _(metadata_df, np, pathlib, pd, scipy, selected_df):
    def latent_from_analysis(file: str | pathlib.Path, return_distances: bool = False):

        file = pathlib.Path(file)

        assert file.exists()
        df = pd.read_parquet(file)

        # as many rows (windows) as in the *metadata* `DataFrame`
        assert len(df) == len(metadata_df)

        z = df.filter(like='z_').values
        z_selected = z[selected_df['# window']]

        # '# window` should yield an actual index (which is also a label in the `DataFrame`)
        assert np.allclose(z_selected, df.filter(like='z_').loc[selected_df['# window']].values)

        distances = scipy.spatial.distance.cdist(z_selected, z)

        # for the sake of convenience and readability
        i_selected = np.arange(len(selected_df))

        # the closest distance is always 0 (to the element itself)...
        assert (distances[i_selected, selected_df['# window']] == 0.).all()

        # ...and hence the $0$-th element in each row should be ignored
        i_closest = distances.argsort(axis=1)[:, 1:]

        if return_distances:
            return (i_closest, distances[i_selected[:, None], i_closest])
        else:
            return i_closest
    return (latent_from_analysis,)


@app.cell
def _(latent_from_analysis, model_dir):
    # i_closest_vae = latent_from_analysis(model_dir / 'vae.parquet')
    i_closest_vae, distances_vae = latent_from_analysis(model_dir / 'vae.parquet', return_distances=True)
    print(i_closest_vae.shape)
    i_closest_vae
    return (i_closest_vae,)


@app.cell
def _():
    # vae_df = pd.read_parquet(model_dir / 'vae.parquet')
    # vae_df.loc[i_closest_vae[0]].filter(like='z_')
    # z = vae_df.filter(like='z_').values
    # z_selected = z[selected_df['# window']]
    # # z_selected[0].shape
    # # z[i_closest_vae[0,0]].shape
    # # np.sqrt(((z_selected[0] - z[i_closest_vae[0,1]])**2).sum())
    return


@app.cell
def _():
    # n_to_keep = 15
    # # i_selected = 0
    # # metadata_df.loc[i_closest_vae[i_selected]]['signal'].unique()[:25]
    # _to_keep = []
    # for _i_sorted_selected in i_closest_vae:
    #     _to_keep.append(metadata_df.loc[_i_sorted_selected]['signal'].unique()[:n_to_keep])
    # vae_selected_signals = np.concatenate(_to_keep)
    # vae_selected_signals
    return


@app.cell
def _(metadata_df, np):
    def select_signals(i_closest: np.ndarray, n: int) -> np.ndarray:
    
        to_keep = []
    
        for i_sorted_selected in i_closest:
            to_keep.append(metadata_df.loc[i_sorted_selected]['signal'].unique()[:n])
    
        return np.concatenate(to_keep)
    return (select_signals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# AR""")
    return


@app.cell
def _(mo, model_dir):
    # mo.output.append(mo.md('AR coefficients'))
    ar_browser = mo.ui.file_browser(
        initial_path=model_dir.parent / 'AR', selection_mode='file', multiple=False, restrict_navigation=True
    )
    # mo.output.append(ar_browser)
    ar_browser
    return (ar_browser,)


@app.cell
def _(ar_browser, mo):
    ar_analysis_file = ar_browser.path()
    mo.stop(ar_analysis_file is None, mo.md("**Choose AR coefficents.**"))
    return (ar_analysis_file,)


@app.cell
def _(ar_analysis_file, latent_from_analysis):
    i_closest_ar = latent_from_analysis(ar_analysis_file)
    i_closest_ar
    return (i_closest_ar,)


@app.cell
def _(i_closest_ar, metadata_df):
    metadata_df.iloc[i_closest_ar[0, :4]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A `DataFrame` with `n_signals_to_keep_from_every_model` selected according to both VAE and AR models.""")
    return


@app.cell
def _(
    i_closest_ar,
    i_closest_vae,
    n_signals_to_keep_from_every_model,
    pd,
    select_signals,
    selected_df,
):
    signals_to_assess_df = pd.concat(
        (pd.DataFrame(
            {'signal': select_signals(i_closest_vae, n_signals_to_keep_from_every_model),
             'Selected by': ['VAE'] * (n_signals_to_keep_from_every_model * len(selected_df))}),
        pd.DataFrame(
            {'signal': select_signals(i_closest_ar, n_signals_to_keep_from_every_model),
             'Selected by': ['AR'] * (n_signals_to_keep_from_every_model * len(selected_df))})
    ))
    print(signals_to_assess_df.shape)
    signals_to_assess_df.head()
    return (signals_to_assess_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Signals that have **not** been selected by any algorithm""")
    return


@app.cell
def _(metadata_df, signals_to_assess_df):
    unselected = set(metadata_df['signal']) - set(signals_to_assess_df['signal'])
    assert len(unselected) < len(set(metadata_df['signal']))
    return (unselected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Signals selected *randomly*, i.e., not selected by any algorithm, are added to the `DataFrame`""")
    return


@app.cell
def _(n_random_signals_to_keep, np, pd, unselected):
    # np.random.default_rng(42).choice(list(unselected),size=n_random_signals_to_keep,replace=False)
    signals_to_assess_df = pd.concat((
        signals_to_assess_df,
        pd.DataFrame(
            {'signal': np.random.default_rng(42).choice(list(unselected),size=n_random_signals_to_keep,replace=False),
             'Selected by': [None] * n_random_signals_to_keep}
        )
    ))
    return (signals_to_assess_df,)


if __name__ == "__main__":
    app.run()
