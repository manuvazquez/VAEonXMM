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
    mo.md(r"""## Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Path""")
    return


@app.cell
def _(mo, pathlib):
    file_browser = mo.ui.file_browser(
        initial_path=pathlib.Path('results/'), selection_mode='directory', multiple=False, restrict_navigation=True
    )
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser):
    model_dir = file_browser.path()
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
    mo.md(r"""The selection that is currently saved""")
    return


@app.cell
def _(pd, selection_saved_file):
    selected_df = pd.read_excel(selection_saved_file)
    selected_df.head()
    return (selected_df,)


@app.cell
def _(metadata_df, np, pathlib, pd, scipy, selected_df):
    def latent_from_analysis(file: str | pathlib.Path):

        file = pathlib.Path(file)

        assert file.exists()
        df = pd.read_parquet(file)
    
        assert len(df) == len(metadata_df)

        z = df.filter(like='z_').values
        z_selected = z[selected_df['# window']]

        assert np.allclose(z_selected, df.filter(like='z_').loc[selected_df['# window']].values)

        distances = scipy.spatial.distance.cdist(z_selected, z)

        assert (distances[np.arange(len(selected_df)), selected_df['# window']] == 0.).all()

        # return distances
    
        # closest element is always iself, so $0$-th element in each row should be ignored
        i_closest = distances.argsort(axis=1)[:, 1:]

        return i_closest
    return (latent_from_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The projection onto the latent space for those particular windows using the VAE model""")
    return


@app.cell
def _(latent_from_analysis, model_dir):
    latent_from_analysis(model_dir / 'vae.parquet')
    return


@app.cell
def _():
    # vae_analysis = model_dir / 'vae.parquet'
    # assert vae_analysis.exists()
    # vae_analysis_df = pd.read_parquet(vae_analysis)
    # assert len(vae_analysis_df) == len(metadata_df)

    # z_vae = vae_analysis_df.filter(like='z_').values
    # z_selected_vae = z_vae[selected_df['# window']]
    # assert np.allclose(z_selected_vae, vae_analysis_df.filter(like='z_').loc[selected_df['# window']].values)
    return


@app.cell
def _():
    # # z = vae_analysis_df.filter(like='z_').values
    # # z_selected = vae_analysis_df.filter(like='z_').loc[selected_df['# window']].values
    # vae_distances = scipy.spatial.distance.cdist(z_selected_vae, z_vae)
    # vae_distances
    return


@app.cell
def _():
    # vae_distances.min(axis=1)
    # assert (vae_distances[np.arange(len(selected_df)), selected_df['# window']] == 0.).all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The closest element is always iself, so $0$-th element in each row should be ignored""")
    return


@app.cell
def _():
    # vae_distances.argsort(axis=1)
    return


@app.cell
def _(mo, model_dir):
    ar_browser = mo.ui.file_browser(
        initial_path=model_dir.parent / 'AR', selection_mode='file', multiple=False, restrict_navigation=True
    )
    ar_browser
    return (ar_browser,)


@app.cell
def _(ar_browser, latent_from_analysis):
    ar_analysis_file = ar_browser.path()
    latent_from_analysis(ar_analysis_file)
    return


if __name__ == "__main__":
    app.run()
