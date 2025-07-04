import marimo

__generated_with = "0.14.9"
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
    n_signals_to_keep = 200
    n_signals_to_keep_from_every_model = 15
    # n_random_signals_to_keep = 80
    output_file_stem = 'signals_to_assess'
    return (
        n_signals_to_keep,
        n_signals_to_keep_from_every_model,
        output_file_stem,
    )


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
    mo.md(r"""## AR""")
    return


@app.cell
def _(mo, model_dir):
    ar_browser = mo.ui.file_browser(
        initial_path=model_dir.parent / 'AR', selection_mode='file', multiple=False, restrict_navigation=True
    )
    ar_browser
    return (ar_browser,)


@app.cell
def _(ar_browser, mo):
    mo.stop(ar_browser.path() is None, mo.md("**Choose AR coefficents.**"))
    ar_analysis_file = ar_browser.path()
    return (ar_analysis_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We keep track of which AR model was used""")
    return


@app.cell
def _(ar_analysis_file, model_dir, output_file_stem):
    with open((model_dir / output_file_stem).with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write(ar_analysis_file.as_posix())
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
    mo.md(r"""# *Reference* windows""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The *windows* belonging to the (selected beforehand) "signals of interest" (only one window is shown). This is only meant to help in choosing, below, one window from each.""")
    return


@app.cell
def _(metadata_df, mo, trapallada):
    signals_of_interest_df = metadata_df[
        metadata_df['signal'].isin(trapallada.data.Labels().signals)
    ]
    mo.hstack((
        signals_of_interest_df.drop_duplicates('source'),
        trapallada.data.Labels().signals_without_prefix
    ))
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


@app.cell
def _(selected_df):
    set(selected_df['signal'])
    return


@app.cell
def _(metadata_df, selected_df):
    len(metadata_df)
    selected_df.index
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A *numpy* `Array` with the indexes of the "eligible windows" (hose belonging to the "reference signals" are not `eligible` unless they are one the above selected)""")
    return


@app.cell
def _(metadata_df, selected_df):
    # windows belonging to the "reference signals" are not `eligible`...
    eligible = ~metadata_df['signal'].isin(set(selected_df['signal']))

    # ...unless they are one the above selected
    eligible.loc[selected_df['# window']] = True

    i_eligible = metadata_df[eligible].index.to_numpy()
    return (i_eligible,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Ranking""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A function to get the closest windows (as given by the passed *analysis* file) to the selected ones **in the latent space**. No window belonging to the same *signal* of one of the selected windows is picked.""")
    return


@app.cell
def _(i_eligible, metadata_df, np, pathlib, pd, scipy, selected_df):
    def rank_windows(file: str | pathlib.Path, return_distances: bool = False):

        file = pathlib.Path(file)

        assert file.exists()
        df = pd.read_parquet(file)

        # as many rows (windows) as in the *metadata* `DataFrame`
        assert len(df) == len(metadata_df)

        z = df.filter(like='z_').values
        z_selected = z[selected_df['# window']]

        # '# window` should yield an actual index (which is also a label in the `DataFrame`)
        assert np.allclose(z_selected, df.filter(like='z_').loc[selected_df['# window']].values)

        # a submatrix with only the "eligible" windows (`z_selected` should be among them)
        z_eligible = z[i_eligible]

        distances = scipy.spatial.distance.cdist(z_selected, z_eligible)

        # for the sake of convenience and readability
        i_selected = np.arange(len(selected_df))

        # we look up the original indexes of selected windows in `i_eligible` (to get their indexes in `z_eligible`)
        i_selected_window_within_eligible = np.where(i_eligible[:, None]==selected_df['# window'].to_numpy())[0]

        # the closest distance is always 0 (to the element itself)...
        assert (distances[i_selected, i_selected_window_within_eligible] == 0.).all()

        # ...and hence the $0$-th element in each row should be ignored
        i_closest = distances.argsort(axis=1)[:, 1:]

        # the indexes in `i_closest` must be mapped back to the original `z` (which is "in sync" with `metadata_df`)
        if return_distances:
            return (i_eligible[i_closest], distances[i_selected[:, None], i_closest])
        else:
            return i_eligible[i_closest]
    return (rank_windows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A function to, given a ranking, select a number of windows belonging to *different* (i.e., unique) signals""")
    return


@app.cell
def _(metadata_df, np):
    def select_windows(i_closest: np.ndarray, n: int) -> list[int]:

        i_to_keep = []

        for i_sorted_selected in i_closest:
            i_to_keep.extend(metadata_df.loc[i_sorted_selected].drop_duplicates('signal')[:n].index)

        return set(i_to_keep)
    return (select_windows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Output""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## VAE""")
    return


@app.cell
def _(model_dir, rank_windows):
    i_closest_vae, distances_vae = rank_windows(model_dir / 'vae.parquet', return_distances=True)
    print(i_closest_vae.shape)
    i_closest_vae
    return (i_closest_vae,)


@app.cell
def _(i_closest_vae, n_signals_to_keep_from_every_model, select_windows):
    i_windows_selected_vae = select_windows(i_closest_vae, n_signals_to_keep_from_every_model)
    return (i_windows_selected_vae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## AR""")
    return


@app.cell
def _(
    ar_analysis_file,
    n_signals_to_keep_from_every_model,
    rank_windows,
    select_windows,
):
    i_closest_ar = rank_windows(ar_analysis_file)
    i_windows_selected_ar = select_windows(i_closest_ar, n_signals_to_keep_from_every_model)
    return (i_windows_selected_ar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Final signals""")
    return


@app.cell
def _(i_windows_selected_ar, i_windows_selected_vae, mo):
    # len(i_windows_selected_vae), len(i_windows_selected_ar)
    mo.md(f'''
    Number of windows selected:

    - VAE: {len(i_windows_selected_vae)}
    - AR: {len(i_windows_selected_ar)}
    ''')
    return


@app.cell
def _(i_windows_selected_ar, i_windows_selected_vae, mo):
    i_windows_selected_both = i_windows_selected_vae & i_windows_selected_ar
    n_windows_selected_by_both = len(i_windows_selected_both)
    mo.md(f'{n_windows_selected_by_both} window(s) selected by both algorithms')
    return (i_windows_selected_both,)


@app.cell
def _(i_windows_selected_ar, i_windows_selected_vae, mo):
    i_windows_selected_any = i_windows_selected_vae | i_windows_selected_ar
    n_windows_selected_by_any = len(i_windows_selected_any)
    mo.md(f'Both models jointly selected {n_windows_selected_by_any} windows')
    return i_windows_selected_any, n_windows_selected_by_any


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A `DataFrame` with `n_signals_to_keep_from_every_model` *windows* selected according to both VAE and AR models. Every algorithm selects windows belonging to different signals, but both algorithms might select the same window or different windows beloning to the same signal.""")
    return


@app.cell
def _(
    i_windows_selected_any,
    i_windows_selected_ar,
    i_windows_selected_both,
    i_windows_selected_vae,
    metadata_df,
    pd,
):
    signals_to_assess_from_models_df = metadata_df.loc[list(i_windows_selected_any)]
    signals_to_assess_from_models_df.loc[list(i_windows_selected_vae), 'Selected by'] = 'VAE'
    signals_to_assess_from_models_df.loc[list(i_windows_selected_ar), 'Selected by'] = 'AR'
    signals_to_assess_from_models_df.loc[list(i_windows_selected_both), 'Selected by'] = 'VAE & AR'
    signals_to_assess_from_models_df['Selected by'] = pd.Categorical(signals_to_assess_from_models_df['Selected by'])
    signals_to_assess_from_models_df
    return (signals_to_assess_from_models_df,)


@app.cell
def _(mo, n_windows_selected_by_any, signals_to_assess_from_models_df):
    mo.md(f"""The {n_windows_selected_by_any} windows belong to {len(signals_to_assess_from_models_df['signal'].unique())} signals""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A `Series` which is `True` for every window whose signal has NOT been selected by one of the algorithms.""")
    return


@app.cell
def _(metadata_df, signals_to_assess_from_models_df):
    signal_not_selected_yet = ~metadata_df['signal'].isin(set(signals_to_assess_from_models_df['signal']))
    signal_not_selected_yet.tail()
    return (signal_not_selected_yet,)


@app.cell
def _(mo, n_signals_to_keep, n_windows_selected_by_any):
    n_random_signals_to_keep = n_signals_to_keep - n_windows_selected_by_any
    mo.md(f'{n_random_signals_to_keep} signals selected *randomly* (so that overall {n_signals_to_keep} are presented), i.e., not selected by any algorithm, are added to the `DataFrame`')
    return (n_random_signals_to_keep,)


@app.cell
def _(
    metadata_df,
    n_random_signals_to_keep,
    pd,
    signal_not_selected_yet,
    signals_to_assess_from_models_df,
):
    random_signals_to_assess_df = (
        metadata_df[
            # windows whose corresponding signals have NOT been selected
            signal_not_selected_yet
        # a single signal is selected from every window
        ].groupby('signal').sample(n=1, random_state=42)
        # the "survivors" from the above filters are sampled randomly
        .sample(n=n_random_signals_to_keep, random_state=43)
    )
    assert set(signals_to_assess_from_models_df['signal']) & set(random_signals_to_assess_df['signal']) == set()

    signals_to_assess_df = pd.concat((signals_to_assess_from_models_df,random_signals_to_assess_df))
    signals_to_assess_df
    return (signals_to_assess_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The above `DataFrame` is *shuffled* saved as it to a *.parquet* file and **without the index** to an *Excel* spreadsheet.""")
    return


@app.cell
def _(model_dir, output_file_stem, signals_to_assess_df):
    shuffled_signals_to_assess_df = signals_to_assess_df.sample(frac=1, random_state=42)
    shuffled_signals_to_assess_df.to_parquet(model_dir / (output_file_stem + '.parquet'))
    shuffled_signals_to_assess_df.drop('Selected by', axis=1).to_excel(model_dir / (output_file_stem + '.xlsx'), index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Importance sampling""")
    return


@app.cell
def _(mo, pd, signal_not_selected_yet):
    is_q_inverse_series = pd.Series({
        'VAE': 1,
        'AR': 1,
        'VAE & AR': 1,
        'None': signal_not_selected_yet.sum()
    })
    is_weight_series = is_q_inverse_series / is_q_inverse_series.sum()
    mo.hstack((is_q_inverse_series, is_weight_series))
    return


if __name__ == "__main__":
    app.run()
