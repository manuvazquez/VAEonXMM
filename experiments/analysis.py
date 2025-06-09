import marimo

__generated_with = "0.13.15"
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

    import pandas as pd

    import plotly
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go

    import trapallada.analysis
    import trapallada.results.analysis
    import trapallada.results.plot
    return go, make_subplots, pathlib, pd, px, trapallada


@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = 'notebook'
    return


@app.cell
def _(mo):
    args = mo.cli_args()
    return (args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The base directory""")
    return


@app.cell
def _(trapallada):
    base_dir = trapallada.analysis.get_base_directory()
    base_dir
    return (base_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Input Files""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Particular *sub*directory with the trained model. It should be a `str` so that parameters can be injected with papermill.""")
    return


@app.cell
def _(args, mo, pathlib):
    if not args:
        file_browser = mo.ui.file_browser(
            initial_path=pathlib.Path('results/'), selection_mode='directory', multiple=False, restrict_navigation=True
        )
        file_browser
    else:
        file_browser = None
    file_browser
    return (file_browser,)


@app.cell
def _(args, file_browser, pathlib):
    if not args:
        model_dir = file_browser.path()
    else:
        model_dir = pathlib.Path(args['model_dir'])
    return (model_dir,)


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
    ds, positions_df, model, training_df, parameters = trapallada.analysis.load_results_dir(model_dir)
    return ds, model, parameters, positions_df, training_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The path to *SIMBAD (XMATCH)* database""")
    return


@app.cell
def _(base_dir):
    catalog_file = base_dir / 'experiments' / 'Cross_Simbad_r3.5.csv'
    assert catalog_file.exists()
    return (catalog_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If the *analysis* `DataFrame` was not built before, it is.""")
    return


@app.cell
def _(catalog_file, ds, model, model_dir, pd, positions_df, trapallada):
    analysis_file = model_dir / 'analysis.parquet'
    if not analysis_file.exists():

        print(f'Making "{analysis_file.name}"...')

        # projecting into the latent space and clustering
        analysis_df = trapallada.results.analysis.cluster(ds, model.encoder)

        # position information is appended
        analysis_df = trapallada.results.analysis.augment_with_positions(analysis_df, positions_df)

        # labels from external database
        analysis_df = trapallada.results.analysis.augment_with_simbad(analysis_df, catalog_file)

        # extra columns with the number of window and signal "id"
        analysis_df = trapallada.results.analysis.augment_with_window_and_signal_ids(analysis_df)

        analysis_df.to_parquet(analysis_file)
    else:
        analysis_df = pd.read_parquet(analysis_file)
    return (analysis_df,)


@app.cell
def _(analysis_df):
    analysis_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For the sake of convenience""")
    return


@app.cell
def _():
    tsne_columns = [f'tsne_{i}' for i in range(1, 3)]
    return (tsne_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The clustering (either *in projected space* or *in z*) to be analyzed""")
    return


@app.cell
def _():
    available_clusterings = ['cluster in z', 'cluster in projected space']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Seemingly, *parquet* format dismisses `Categorical` data type""")
    return


@app.cell
def _(analysis_df):
    available_clusterings_1 = ['cluster in z', 'cluster in projected space']
    analysis_df[available_clusterings_1].dtypes
    return (available_clusterings_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It is manually fixed""")
    return


@app.cell
def _(analysis_df, available_clusterings_1):
    for _c in available_clusterings_1:
        analysis_df[_c] = analysis_df[_c].astype('category')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The number of objects for every *type*""")
    return


@app.cell
def _(analysis_df):
    analysis_df['main_type'].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""Columns that will be shown when hovering over elements in the plot""")
    return


@app.cell
def _(available_clusterings_1):
    on_plot_hover_columns = ['index', 'source', 'main_type'] + available_clusterings_1
    on_plot_hover_columns
    return (on_plot_hover_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Some metadata on the XMM-Newton""")
    return


@app.cell
def _(trapallada):
    xmm_newton_metadata = trapallada.analysis.get_xmm_newton_metadata()
    return (xmm_newton_metadata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Training""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Curves for the training/validation

    - loss, 
    - log-likelihood, and
    - KL divergence
    """
    )
    return


@app.cell
def _(training_df, trapallada):
    trapallada.analysis.training_chart(training_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Number of parameters in the model""")
    return


@app.cell
def _(model, trapallada):
    sum(trapallada.util.get_layers_n_parameters(model).values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If using a CVAE, the convolutional kernel in the decoder""")
    return


@app.cell
def _(model):
    try:
        print(model.decoder.conv_layer.weight)
    except AttributeError:
        print('no convolutional layer in the decoder')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Latent space""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mean and standard deviation for every variable in the latent space""")
    return


@app.cell
def _(analysis_df, px):
    latents_df = analysis_df.filter(like='z_').agg(['mean', 'std']).T
    # latents_df.head()
    px.scatter(latents_df, x=latents_df.index, y='mean', error_y='std')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Javi's paper""")
    return


@app.cell
def _(ds, model, trapallada, xmm_newton_metadata):
    # trapallada.analysis.plot_fit_file_window(model, ds, 'P0605950101PNS003SRCTSR8001.FTZ', -1)
    trapallada.analysis.plot_fit_file_window(model, ds, xmm_newton_metadata['fit files']['javi'][0], -1)
    return


@app.cell
def _(ds, model, trapallada, xmm_newton_metadata):
    # trapallada.analysis.plot_fit_file_window(model, ds, 'P0605950101PNS003SRCTSR8003.FTZ')
    trapallada.analysis.plot_fit_file_window(model, ds, xmm_newton_metadata['fit files']['javi'][1], -1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Clusters""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Clusters over

    - the latent space, and
    - *projected space* (t-SNE).

    **CAVEAT**: they are not necessarily computed with the same parameters.
    """
    )
    return


@app.cell
def _(analysis_df, available_clusterings_1, go, make_subplots):
    _fig = make_subplots(rows=1, cols=2, subplot_titles=tuple(available_clusterings_1))
    for i_c, _c in enumerate(available_clusterings_1):
        n_samples_per_cluster = analysis_df[_c].value_counts()
        n_clusters = len(n_samples_per_cluster)
        _fig.add_trace(go.Bar(x=n_samples_per_cluster.index, y=n_samples_per_cluster.values, name=_c), row=1, col=i_c + 1)
        print(f'{n_clusters} {_c}')
    _fig.update_layout(title_text='Clusters', showlegend=False)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The directory with the full signals is inferred""")
    return


@app.cell
def _(parameters, pathlib):
    full_signals_dir = (pathlib.Path.home() / parameters['dataset']['files']['signals_file']).parent / 'full_signals'
    full_signals_dir
    return (full_signals_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The windows in the projected space, colored according to its

    - cluster in the latent space (`z`)
    """
    )
    return


@app.cell
def _(
    analysis_df,
    full_signals_dir,
    mo,
    on_plot_hover_columns,
    trapallada,
    tsne_columns,
):
    plot = mo.ui.plotly(
        trapallada.analysis.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns).make_fig(color_column='cluster in z')
    )
    plot
    return (plot,)


@app.cell
def _(analysis_df, plot):
    analysis_df.loc[plot.indices]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""(some signals)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- cluster in the projected space""")
    return


@app.cell
def _(
    analysis_df,
    full_signals_dir,
    on_plot_hover_columns,
    trapallada,
    tsne_columns,
):
    trapallada.analysis.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns).make_fig(color_column='cluster in projected space')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""(some signals)""")
    return


@app.cell
def _():
    # trapallada.analysis.plot_samples_from_cluster(analysis_df, full_signals_dir, cluster_column='cluster in projected space')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Signals by `main_type`""")
    return


@app.cell
def _():
    re_drop_columns= '^(?!z_|tsne_|set$|RA|DEC|CRA|CDEC|signal)'
    return (re_drop_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The (selected beforehand) *signals of interest*""")
    return


@app.cell
def _(analysis_df, trapallada):
    signals_of_interest_df = analysis_df[
        analysis_df['signal'].isin(trapallada.analysis.XMMNewtonMetadata().signals)
    ]
    # signals_of_interest_df.drop_duplicates('source').filter(regex=re_drop_columns)
    signals_of_interest_df.drop_duplicates('source')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Empty `main_type`s are set to `UNKNOWN`""")
    return


@app.cell
def _(analysis_df):
    analysis_df['main_type'] = analysis_df['main_type'].fillna('UNKNOWN')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Assessment""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In order to have fixed axes""")
    return


@app.cell
def _(analysis_df, tsne_columns):
    xrange, yrange = list(zip(analysis_df[tsne_columns].min(), analysis_df[tsne_columns].max()))
    # xrange, yrange
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    All the windows are plotted, and some of them, considered *of interest*, are highlighted.

    One can **double-click** a `main_type` in the *Legend* to only show the corresponding curves
    """
    )
    return


@app.cell
def _(
    analysis_df,
    full_signals_dir,
    mo,
    on_plot_hover_columns,
    trapallada,
    tsne_columns,
):
    tsne_plot = trapallada.analysis.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns)
    _fig = tsne_plot.make_fig()
    tsne_plot.highlight_windows_of_interest()
    # _fig.show()
    mo_plot = mo.ui.plotly(_fig)
    mo_plot
    return (mo_plot,)


@app.cell
def _(analysis_df, mo_plot):
    max_n_samples_to_show = 30
    i_selected = [e['index'] for e in mo_plot.points if e]
    selected_samples = analysis_df.loc[i_selected]
    if len(selected_samples) > max_n_samples_to_show:
        selected_samples = selected_samples.drop_duplicates(subset=['signal'])
        print('Too many selected, plotting just one window for every signal...')
        if len(selected_samples) > max_n_samples_to_show:
            selected_samples = selected_samples.sample(max_n_samples_to_show)
            print(f'Still too many, plotting {max_n_samples_to_show} randomly selected...')
    return (selected_samples,)


@app.cell
def _(
    full_signals_dir,
    mo,
    model,
    re_drop_columns,
    selected_samples,
    trapallada,
):
    for i_row, row in selected_samples.iterrows():
        mo.output.append(row.filter(regex=re_drop_columns).to_frame().T)
        signal_id, *start_end = row[['signal', 'start', 'end']]
        pred = model.decoder(selected_samples.filter(like='z_').loc[i_row])
        # mo.output.append(trapallada.results.plot.full_signals(signal_id, full_signals_dir, *start_end))
        mo.output.append(trapallada.results.plot.full_signals(signal_id, full_signals_dir, *start_end, pred))
        mo.output.append('---')
    return


if __name__ == "__main__":
    app.run()
