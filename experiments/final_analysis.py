import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Te mando la lista de correlación entre tu lista y la base de datos. Le he puesto que busque en un radio de 5 arcsec de cada objeto de tu lista, así que, para alguno de ellos, hay múltiples candidatos. Si alguno de esos nos interesa, yo elegiría el primero que aparezca, que será el más cercano.

    En esta lista que te mando, he identificado algunos “main_type” que nos interesan. En concreto, creo que los interesantes son:

    1.  OrionV*
    2.  SB*
    3.  EclBin
    4.  TTauri*
    5.  Variable*
    6.  Planet:  ————>  Creo que sólo hay uno, pero es una de las estrellas con exoplanetas más estudiadas.
    7.  BYDraV*
    8.  Or*:   —————>  Esta es la misma que OrionV*, no sé por qué usan ambas nomenclaturas.
    9.  Catacly*
    10. YSO
    11. Eruptive*
    12. Transient
    13. RSCVnV*
    14. Low-Mass*


    Y, luego, hay muchas que son bastante raras y merecería la pena mirar una a una.

    Cuando tengas un rato, con calma, yo te diría de sacar un pdf de cada uno de los objetos de tu lista, que incluya todas las series temporales que tengas de ese objeto (me refiero a cada división de serie temporal que hayas hecho para esa observación), marcando de algún modo (quizás de otro color) la serie temporal que se corresponda al punto que tú has seleccionado (porque sabemos que la posición en el diagrama depende de si se ve la característica o no en tu selección de 20 ks, creo que era) y el nombre del objeto identificado en mi lista, con el main_type y las coordenadas, si quieres. Sería una buena forma de poder ver toda tu selección, sus series temporales y cuál es su naturaleza física, según el cross-match que yo he hecho.

    Alternativamente, puedes seleccionar sólo las que tengan alguno de los 14 tipos que te indico y pintar su serie temporal e identificad el main_type y coordenadas. Lo que te he dicho antes es para aprovechar, si lo haces automático, y comprobar esos tipos raros que he encontrado.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. El tipo 8, Or*, en realidad no está escrito en main_type, sino en otra columna. En main_type aparece siembre OrionV*. Puedes quedarte sólo con ese tipo y no buscar Or*.

    2. Tú también tienes múltiples coincidencias. Se ve que la estrella HD 189733, que tiene un exoplaneta, debe de tener una fulguración medio larga en una observación y, al dividir la serie temporal, al menos dos de las subdivisiones de 20 ks con las que trabajas han caído en la zona buena. Ademas, tiene otra observación posterior donde también sale. De hecho, esta estrella la estudiamos en un artículo hace 3 o 4 años y tenemos todas sus series temporales ahí, y se ven todas las fulguraciones que tuvo. Nos viene bien como benchmark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Check the signals in some clusters
    """)
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
    import trapallada.util

    return pathlib, pd, trapallada


@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = 'notebook'
    return


@app.cell
def _(mo):
    args = mo.cli_args()
    args
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The base directory
    """)
    return


@app.cell
def _(trapallada):
    base_dir = trapallada.util.get_base_directory()
    base_dir
    return (base_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parameters
    """)
    return


@app.cell
def _():
    max_n_samples_to_show = 5

    # for t-SNE
    n_projected_components = 2
    return max_n_samples_to_show, n_projected_components


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell
def _():
    algorithm = 'vae'
    return (algorithm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _(base_dir):
    model_dir = base_dir / 'results' / 'paper_main'
    return (model_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Analysis data* produced here can be associated with a particular trained model (for the case of VAEs) or with the `Dataset` (for the case of the AR model). The name of the file is also dependent on the model.
    """)
    return


@app.cell
def _(algorithm, model_dir):
    analysis_file_stem = algorithm
    analysis_dir = model_dir

    analysis_dir
    return analysis_dir, analysis_file_stem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Input Files
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Particular *sub*directory with the trained model. It should be a `str` so that parameters can be injected with papermill.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the results directory,

    - the `Dataset`,
    - the headers of the corresponding files,
    - the ML model,
    - a `DataFrame` with results from training,
    - a `dict` with the parameters

    are obtained
    """)
    return


@app.cell
def _(model_dir, trapallada):
    ds, positions_df, model, training_df, parameters = trapallada.results.analysis.load_dir(model_dir)
    return model, parameters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dimension of the latent space can be inferred from the *encoder*
    """)
    return


@app.cell
def _(model):
    dim_z = model.encoder.get_latent_dim()
    dim_z
    return (dim_z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The names of the columns for the latent space projections (to be added to a `DataFrame`
    ). `z_sd_cols` are only available for VAE models.
    """)
    return


@app.cell
def _(dim_z, mo):
    z_cols = [f'z_{i}' for i in range(1, dim_z + 1)]
    z_sd_cols = [f'z_sd_{i}' for i in range(1, dim_z+1)]
    z_cols, z_sd_cols
    mo.hstack((z_cols, z_sd_cols))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For t-SNE
    """)
    return


@app.cell
def _(n_projected_components):
    tsne_cols = [f'tsne_{i}' for i in range(1, n_projected_components+1)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the *analysis* `DataFrame` was not built before, it is.
    """)
    return


@app.cell
def _(analysis_dir, analysis_file_stem, pd):
    analysis_file = analysis_dir / f'{analysis_file_stem}.parquet'
    analysis_df = pd.read_parquet(analysis_file)
    print(f'reading "{analysis_file}"')
    return (analysis_df,)


@app.cell
def _(analysis_df):
    analysis_df.head()
    return


@app.cell
def _(analysis_df):
    analysis_df['wasserstein cluster in z']
    return


@app.cell
def _(analysis_df):
    analysis_df.filter(like='z_').shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the sake of convenience
    """)
    return


@app.cell
def _():
    tsne_columns = [f'tsne_{i}' for i in range(1, 3)]
    return (tsne_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The clustering (either *in projected space* or *in z*) to be analyzed
    """)
    return


@app.cell
def _(algorithm):
    available_clusterings = ['cluster in z', 'cluster in projected space']

    if algorithm == 'vae':
        available_clusterings += ['wasserstein cluster in z']
    return (available_clusterings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Seemingly, *parquet* format dismisses `Categorical` data type
    """)
    return


@app.cell
def _(analysis_df, available_clusterings):
    analysis_df[available_clusterings].dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is manually fixed
    """)
    return


@app.cell
def _(analysis_df, available_clusterings):
    for cluster in available_clusterings:
        analysis_df[cluster] = analysis_df[cluster].astype('category')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The number of objects for every *type*
    """)
    return


@app.cell
def _(analysis_df):
    analysis_df['main_type'].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Columns that will be shown when hovering over elements in the plot
    """)
    return


@app.cell
def _(available_clusterings):
    on_plot_hover_columns = ['index', 'source', 'main_type'] + available_clusterings
    on_plot_hover_columns
    return (on_plot_hover_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The directory with the full signals is inferred
    """)
    return


@app.cell
def _(parameters, pathlib):
    full_signals_dir = (pathlib.Path.home() / parameters['dataset']['files']['signals_file']).parent / 'full_signals'
    full_signals_dir
    return (full_signals_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The (selected beforehand) *signals of interest*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Empty `main_type`s are set to `UNKNOWN`
    """)
    return


@app.cell
def _(analysis_df):
    analysis_df['main_type'] = analysis_df['main_type'].fillna('UNKNOWN')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assessment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to have fixed axes
    """)
    return


@app.cell
def _(analysis_df, tsne_columns):
    xrange, yrange = list(zip(analysis_df[tsne_columns].min(), analysis_df[tsne_columns].max()))
    # xrange, yrange
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All the windows are plotted, and some of them, considered *of interest* ("reference signals" in the paper), are highlighted.

    One can **double-click** a `main_type` in the *Legend* to only show the corresponding curves
    """)
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
    mo.md(r"""
    Some filtering of the selection
    """)
    return


@app.cell
def _(
    algorithm,
    analysis_df,
    full_signals_dir,
    max_n_samples_to_show,
    mo_plot,
    model,
    trapallada,
):
    i_selected = [e['index'] for e in mo_plot.points if e and 'index' in e]

    # the output is captured in `_` so that what the function returns doesn't become the final output of the cell (instead of whatever the function produces)
    _ = trapallada.results.plot.marimo_selected(i_selected, analysis_df, model, algorithm, full_signals_dir, max_n_samples_to_show)
    return (i_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For Javi to analyze with SIMBAD
    """)
    return


@app.cell
def _(analysis_df, i_selected):
    # analysis_df.loc[i_selected]['signal']
    analysis_df.loc[i_selected, ['RA', 'DEC', 'CRA', 'CDEC', 'signal']].drop_duplicates(subset='signal')
    return


if __name__ == "__main__":
    app.run()
