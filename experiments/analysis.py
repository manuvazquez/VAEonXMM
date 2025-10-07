import marimo

__generated_with = "0.16.5"
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
    import os
    import pathlib

    import pandas as pd

    import yaml

    from torch.utils.data import DataLoader

    import plotly
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go

    import trapallada.analysis
    import trapallada.results.analysis
    import trapallada.results.plot
    import trapallada.model.ar
    return DataLoader, go, make_subplots, pathlib, pd, px, trapallada


@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = 'notebook'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Parameters""")
    return


@app.cell
def _():
    max_n_samples_to_show = 30

    # for t-SNE
    n_projected_components = 2
    return max_n_samples_to_show, n_projected_components


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model""")
    return


@app.cell
def _(mo):
    radio = mo.ui.radio(
        options={"VAE": 'vae', "AR": 'ar'},
        value="VAE",
        label="Model",
    )
    radio
    return (radio,)


@app.cell
def _(mo, radio):
    if radio.value == 'ar':
        slider = mo.ui.form(element=mo.ui.slider(start=2, stop=20, label="AR process order", value=5))
        mo.output.append(slider)
    return (slider,)


@app.cell
def _(mo, radio, slider):
    if radio.value == 'ar':
        mo.stop(slider.value is None, mo.md("**Choose the order of the AR model**"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data""")
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
    mo.stop(model_dir is None, mo.md("**Choose a path.**"))
    return (model_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Analysis data* produced here can be associated with a particular trained model (for the case of VAEs) or with the `Dataset` (for the case of the AR model). The name of the file is also dependent on the model.""")
    return


@app.cell
def _(model_dir, parameters, radio, slider):
    analysis_file_stem = radio.value

    if radio.value == 'vae':
        analysis_dir = model_dir
    else:
        analysis_dir = model_dir.parent / 'AR'
        analysis_dir.mkdir(exist_ok=True)

        analysis_file_stem += f'_{slider.value}' + '_' + '_'.join([f'{k}_{v}' for k, v in parameters['data'].items()])

    analysis_dir
    return analysis_dir, analysis_file_stem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Input Files""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The base directory""")
    return


@app.cell
def _(trapallada):
    base_dir = trapallada.util.get_base_directory()
    base_dir
    return (base_dir,)


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
    return ds, model, parameters, positions_df, training_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The dimension of the latent space can be inferred from the *encoder*""")
    return


@app.cell
def _(model):
    dim_z = model.encoder.get_latent_dim()
    dim_z
    return (dim_z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The names of the columns for the latent space projections (to be added to a `DataFrame`
    ). `z_sd_cols` are only available for VAE models.
    """
    )
    return


@app.cell
def _(dim_z, ds, mo, radio, slider):
    # z_cols = [f'z_{i}' for i in range(1, dim_z + 1)]
    if radio.value == 'vae':
        z_cols = [f'z_{i}' for i in range(1, dim_z + 1)]
    else:
        mo.stop(slider.value is None, mo.md("**Choose the order of the AR model**"))
        z_cols = [f'z_{i}' for i in range(1, ds[0][1].shape[0]**2 * slider.value + 1)]

    z_sd_cols = [f'z_sd_{i}' for i in range(1, dim_z+1)]
    z_cols, z_sd_cols
    mo.hstack((z_cols, z_sd_cols))
    return z_cols, z_sd_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For t-SNE""")
    return


@app.cell
def _(n_projected_components):
    tsne_cols = [f'tsne_{i}' for i in range(1, n_projected_components+1)]
    return (tsne_cols,)


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
def _(
    DataLoader,
    analysis_dir,
    analysis_file_stem,
    catalog_file,
    ds,
    mo,
    model,
    n_projected_components,
    pd,
    positions_df,
    radio,
    slider,
    trapallada,
    tsne_cols,
    z_cols,
    z_sd_cols,
):
    analysis_file = analysis_dir / f'{analysis_file_stem}.parquet'
    if not analysis_file.exists():

        print(f'Making "{analysis_file.name}"...')

        # a `DataFrame` with metadata
        analysis_df = ds.metadata_df.copy()

        # a `DataLoader` for the `Dataset`
        dl = DataLoader(ds, batch_size=128, shuffle=False)

        # VAE
        if radio.value == 'vae':

            # projection onto the latent space...
            z, z_sd = trapallada.results.analysis.project_onto_latent_space(dl, model.encoder, return_sd=True)

            analysis_df[z_sd_cols] = z_sd

            # cluster in the latent space using Wasserstein distance
            analysis_df['wasserstein cluster in z'] = pd.Categorical(trapallada.results.analysis.wasserstein_cluster(z, z_sd))

        # AR
        else:

            mo.stop(slider.value is None, mo.md("**Choose the order of the AR model**"))

            # the "latent space" is given by the AR coefficients
            z = trapallada.model.ar.ols_dataset_fit(ds, p=slider.value)

        # the above latent spaceo is projected, for visualization, using t-SNE
        tsne = trapallada.results.analysis.tsne_project(z, n_projected_components=n_projected_components)

        analysis_df[z_cols] = z
        analysis_df[tsne_cols] = tsne

        # clustering with HDBSCAN
        analysis_df['cluster in z'] = trapallada.results.analysis.hdbscan_cluster(z, 10)
        analysis_df['cluster in projected space'] = trapallada.results.analysis.hdbscan_cluster(tsne, 80)

        # position information is appended
        analysis_df = trapallada.results.analysis.augment_with_positions(analysis_df, positions_df)

        # labels from external database
        analysis_df = trapallada.results.analysis.augment_with_simbad(analysis_df, catalog_file)

        # extra columns with the number of window and signal "id"
        analysis_df = trapallada.results.analysis.augment_with_window_and_signal_ids(analysis_df)

        analysis_df.to_parquet(analysis_file)
    else:
        analysis_df = pd.read_parquet(analysis_file)
        print(f'reading "{analysis_file}"')
    return (analysis_df,)


@app.cell
def _(analysis_df):
    analysis_df.head()
    return


@app.cell
def _(analysis_df, radio):
    if radio.value == 'vae':
        analysis_df['wasserstein cluster in z']
    return


@app.cell
def _(analysis_df):
    analysis_df.filter(like='z_').shape
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
def _(radio):
    available_clusterings = ['cluster in z', 'cluster in projected space']

    if radio.value == 'vae':
        available_clusterings += ['wasserstein cluster in z']
    return (available_clusterings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Seemingly, *parquet* format dismisses `Categorical` data type""")
    return


@app.cell
def _(analysis_df, available_clusterings, radio):
    if radio.value == 'vae':
        analysis_df[available_clusterings].dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It is manually fixed""")
    return


@app.cell
def _(analysis_df, available_clusterings):
    for cluster in available_clusterings:
        analysis_df[cluster] = analysis_df[cluster].astype('category')
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
def _(available_clusterings):
    on_plot_hover_columns = ['index', 'source', 'main_type'] + available_clusterings
    on_plot_hover_columns
    return (on_plot_hover_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Some metadata on the XMM-Newton""")
    return


@app.cell
def _(trapallada):
    # xmm_newton_metadata = trapallada.analysis.get_xmm_newton_metadata()
    xmm_newton_metadata = trapallada.data.Labels(
        **{k: v for k,v in trapallada.data.labels_files().items() if k == 'reference_signals_file'}
    ).reference_signals_data
    xmm_newton_metadata
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
def _(mo, radio, training_df, trapallada):
    if radio.value == 'vae':
        mo.output.append(trapallada.analysis.training_chart(training_df))
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
    mo.md(r"""## Mean""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mean and standard deviation for every **mean** in the latent space""")
    return


@app.cell
def _(analysis_df, px):
    latents_df = analysis_df.filter(regex=r'z_\d+').agg(['mean', 'std']).T
    px.scatter(latents_df, x=latents_df.index, y='mean', error_y='std')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Standard deviation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mean and standard deviation for every **standar deviation** in the latent space""")
    return


@app.cell
def _(analysis_df, mo, px, radio):
    if radio.value == 'vae':
        latents_sd_df = analysis_df.filter(regex=r'z_sd_\d+').agg(['mean', 'std']).T
        mo.output.append(px.scatter(latents_sd_df, x=latents_sd_df.index, y='mean', error_y='std'))
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
def _(analysis_df, available_clusterings, go, make_subplots):
    fig = make_subplots(rows=1, cols=len(available_clusterings), subplot_titles=tuple(available_clusterings))
    for i_c, c in enumerate(available_clusterings):
        n_samples_per_cluster = analysis_df[c].value_counts()
        n_clusters = len(n_samples_per_cluster)
        fig.add_trace(go.Bar(x=n_samples_per_cluster.index, y=n_samples_per_cluster.values, name=c), row=1, col=i_c + 1)
        print(f'{n_clusters} {c}')
    fig.update_layout(title_text='Clusters', showlegend=False)
    fig
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
    on_plot_hover_columns,
    trapallada,
    tsne_columns,
):
    trapallada.results.plot.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns).make_fig(color_column='cluster in z')
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
    trapallada.results.plot.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns).make_fig(color_column='cluster in projected space')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- cluster in the wasserstein space""")
    return


@app.cell
def _(
    analysis_df,
    full_signals_dir,
    mo,
    on_plot_hover_columns,
    radio,
    trapallada,
    tsne_columns,
):
    if radio.value == 'vae':
        mo.output.append(trapallada.results.plot.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns).make_fig(color_column='wasserstein cluster in z'))
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The (selected beforehand) *signals of interest*""")
    return


@app.cell
def _(analysis_df, trapallada):
    # reference_signals = trapallada.data.Labels(reference_signals_file=trapallada.data.labels_files()['reference_signals_file']).signals
    signals_of_interest_df = analysis_df[
        analysis_df['signal'].isin(trapallada.data.Labels().signals)
    ]
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
    tsne_plot = trapallada.results.plot.TSNE(analysis_df, tsne_columns, full_signals_dir, on_plot_hover_columns)
    tsne_plot_fig = tsne_plot.make_fig()
    tsne_plot.highlight_windows_of_interest()
    mo_plot = mo.ui.plotly(tsne_plot_fig)
    mo_plot
    return (mo_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plots are saved""")
    return


@app.cell
def _():
    # output_dir = pathlib.Path.cwd() / 'plots' / model_dir.name
    # output_dir.mkdir(exist_ok=True)

    # # scatterplot
    # tsne_plot_fig.write_image(output_dir / 'tSNE.pdf')

    # scatter_df = analysis_df[tsne_columns].copy()
    # scatter_df['reference'] = 0

    # for indexes, _ in trapallada.results.analysis.i_windows_of_interest(analysis_df):
    #     scatter_df.loc[indexes, 'reference'] = 1

    # scatter_df.to_csv(output_dir / 'tSNE.csv', index=False)

    # # range
    # if mo_plot.ranges != {}:
    #     with open(f'{output_dir / f'tSNE_region'}.yaml', 'w') as f:
    #         yaml.dump(mo_plot.ranges, f)

    # os.symlink(output_dir.parent / 'reference_vs_non-reference.tex', output_dir / (model_dir.name + '.tex'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Some filtering of the selection""")
    return


@app.cell
def _(
    analysis_df,
    full_signals_dir,
    max_n_samples_to_show,
    mo_plot,
    model,
    radio,
    trapallada,
):
    trapallada.results.plot.marimo_selected(mo_plot, analysis_df, model, radio.value, full_signals_dir, max_n_samples_to_show)
    return


if __name__ == "__main__":
    app.run()
