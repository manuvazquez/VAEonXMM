import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> Compare the performances of VAE and AR in identifying events of interes""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pathlib
    from collections import namedtuple

    import pandas as pd

    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly import express as px
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.metrics import RocCurveDisplay
    import matplotlib.figure

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from jaxtyping import Array, Float, Int

    from torch.utils.data import DataLoader

    import trapallada.util
    import trapallada.results.analysis
    import trapallada.model.training
    import trapallada.model.nn
    return (
        Array,
        EmpiricalCovariance,
        Float,
        Int,
        auc,
        average_precision_score,
        namedtuple,
        pd,
        precision_recall_curve,
        px,
        roc_auc_score,
        roc_curve,
        trapallada,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""So that *plotly*'s pictures are correctly rendered""")
    return


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
    mo.md(r"""Base directory""")
    return


@app.cell
def _(trapallada):
    base_dir = trapallada.util.get_base_directory()
    assert base_dir.exists()
    return (base_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results directory""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Directory with all the files""")
    return


@app.cell
def _(base_dir, mo):
    file_browser = mo.ui.file_browser(
        initial_path=base_dir / 'results', selection_mode='directory', multiple=False, restrict_navigation=True
    )
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, mo):
    res_dir = file_browser.path()
    mo.stop(res_dir is None, mo.md("**Choose a path.**"))
    return (res_dir,)


@app.cell
def _(mo, res_dir):
    vae_analysis_file = res_dir / 'vae.parquet'
    mo.stop(not vae_analysis_file.exists(), mo.md("**Analysis for VAE must be run.**"))
    return (vae_analysis_file,)


@app.cell
def _(mo, res_dir):
    ar_analysis_file = res_dir.parent / 'AR' / 'ar_5_fft_False.parquet'
    mo.stop(not ar_analysis_file.exists(), mo.md("**Analysis for AR must be run.**"))
    return (ar_analysis_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Sub*directory with the labels""")
    return


@app.cell
def _(base_dir):
    annotations_dir = base_dir / 'annotations'
    assert annotations_dir.exists()
    return (annotations_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The "selected by" file indicating the algorithm (VAE or AR) that selected every window.""")
    return


@app.cell
def _(annotations_dir):
    selected_by_file = annotations_dir / 'signals_to_assess.parquet'
    assert selected_by_file.exists()
    return (selected_by_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The file with the annotations provided by Javi López Santiago.""")
    return


@app.cell
def _(annotations_dir):
    annotations_file = annotations_dir / 'windows.xlsx'
    assert annotations_file.exists()
    return (annotations_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Output file *merging* the above files""")
    return


@app.cell
def _(annotations_dir):
    output_file = annotations_dir / 'annotated_windows.parquet'
    return (output_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Windows* that were manually selected as representative of *signals* that were, in the beginning, marked as interesting.""")
    return


@app.cell
def _(annotations_dir):
    reference_windows_file = annotations_dir / 'reference_windows.xlsx'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Selected windows""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The *selected by* `DataFrame` stating the algorithm (VAE or AR) that selected every window.""")
    return


@app.cell
def _(pd, selected_by_file):
    selected_by_df = pd.read_parquet(selected_by_file)
    selected_by_df
    return (selected_by_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `DataFrame` with the labels provided by the annotator. Its index (window *id*) must be the same as that of the above `DataFrame`. The label `Interesting` actually stands for something that has been identified, but it might just be `background` (i.e., noise).""")
    return


@app.cell
def _(annotations_file, pd, selected_by_df):
    # labels_df = pd.read_excel(labels_file)
    labels_df = pd.read_excel(annotations_file)


    # first column is just a "dummy" index (the number of the row)
    labels_df = labels_df.drop(columns=labels_df.columns[0])

    labels_df.index = selected_by_df.index
    labels_df
    return (labels_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Annotations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `DataFrame`s with the labels and the `Selected by` column are merged together...""")
    return


@app.cell
def _(labels_df, pd, selected_by_df):
    res_df = pd.concat((selected_by_df, labels_df), axis=1)
    res_df
    return (res_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...and saved""")
    return


@app.cell
def _(output_file, res_df):
    res_df.to_parquet(output_file, index=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A new column which is `True` for every *event of interest*. `Nan`s are considered *non-event*s.""")
    return


@app.cell
def _(res_df):
    res_df['Event'] = res_df.eval('Label == "Interesting" & Type != "background"')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How many windows of each label, including **unlabeled**""")
    return


@app.cell
def _(res_df):
    res_df['Label'].value_counts(dropna=False)
    return


@app.cell
def _(mo):
    mo.md(r"""*Events* vs *non-events*""")
    return


@app.cell
def _(res_df):
    res_df['Event'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let us get rid of the undecided (`NaN`s)""")
    return


@app.cell
def _(pd, res_df):
    no_undecided_res_df = res_df[~pd.isna(res_df['Label'])]
    no_undecided_res_df.shape
    return (no_undecided_res_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How many windows of each `Type`? Notice the `Type` of the `NOT interesting` is `NaN`""")
    return


@app.cell
def _(no_undecided_res_df):
    no_undecided_res_df['Type'].value_counts(dropna=False).plot(kind='bar')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The *sub*`DataFrame`s for VAE...""")
    return


@app.cell
def _(no_undecided_res_df):
    vae_selected_df = no_undecided_res_df[no_undecided_res_df['Selected by'] == 'VAE'].drop(columns=['Selected by'])
    return (vae_selected_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""", AR, ...""")
    return


@app.cell
def _(no_undecided_res_df):
    ar_selected_df = no_undecided_res_df[no_undecided_res_df['Selected by'] == 'AR'].drop(columns=['Selected by'])
    return (ar_selected_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...and *random*""")
    return


@app.cell
def _(no_undecided_res_df, pd):
    random_selected_df = no_undecided_res_df[pd.isna(no_undecided_res_df['Selected by'])].drop(columns=['Selected by'])
    random_selected_df.shape
    return (random_selected_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Another *sub*`DataFrame` with those that were annotated as `Interesting` (the latter encompass signals identified as *background*)""")
    return


@app.cell
def _(no_undecided_res_df):
    interesting_df = no_undecided_res_df[no_undecided_res_df['Label'] == 'Interesting']
    interesting_df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Precision""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Precision* is computed dismissing "interesting" events that are *background*. Overall precision *in the selection*: out of the total number of selected windows (by any of the algorithms or randomly), how many are interesting (not background).""")
    return


@app.cell
def _(no_undecided_res_df):
    overall_precision = no_undecided_res_df['Event'].mean().item()
    overall_precision
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## VAE""")
    return


@app.cell
def _(vae_selected_df):
    vae_precision = vae_selected_df['Event'].mean().item()
    vae_precision
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## AR""")
    return


@app.cell
def _(ar_selected_df):
    ar_precision = ar_selected_df['Event'].mean().item()
    ar_precision
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Random""")
    return


@app.cell
def _(random_selected_df):
    random_precision = random_selected_df['Event'].mean().item()
    random_precision
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Latent space""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## VAE""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The `DataFrame` with the latent space (embeddings) obtained from the VAE""")
    return


@app.cell
def _(pd, vae_analysis_file):
    vae_df = pd.read_parquet(vae_analysis_file)
    vae_df.head()
    return (vae_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It includes mean and standard deviation""")
    return


@app.cell
def _(vae_df):
    z_mean = vae_df.filter(regex=r'z_\d+').values
    z_sd = vae_df.filter(regex=r'z_sd_\d+').values
    z_mean.shape, z_sd.shape
    return (z_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Mean and standard deviation are stacked together""")
    return


@app.cell
def _():
    # z_mean_sd = jnp.hstack([z_mean, z_sd])
    # z_mean_sd.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## AR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The results for the AR model""")
    return


@app.cell
def _(ar_analysis_file, pd):
    ar_df = pd.read_parquet(ar_analysis_file)
    ar_df.head()
    return (ar_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mahalanobis distance-based comparison""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A function to compare the methods based on Mahalanobis distance (through an [estimated covariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance.mahalanobis)) and different binary classification metrics. This function assumes **the larger the distance from the mean, the more likely a point (i.e., a window) is to be an event of interest**.""")
    return


@app.cell
def _(
    Array,
    EmpiricalCovariance,
    Float,
    Int,
    auc,
    average_precision_score,
    namedtuple,
    precision_recall_curve,
    px,
    roc_auc_score,
    roc_curve,
):
    # so that named values are returned
    MahalanobisBasedMetrics = namedtuple('MahalanobisBasedMetrics', ['average_precision', 'best_recall_at_min_precision', 'roc_curve'])

    def mahalanobis_based_metrics(
            x: Float[Array, 'window feature'],
            labels: Int[Array, 'window'],
            i_windows_of_interest: Int[Array, 'window'],
            min_precision: float = 0.7,
            algorithm_name: str | None = None,
        # ) -> tuple[float, float, float, matplotlib.figure.Figure]:
        ) -> MahalanobisBasedMetrics:

        # covariance matrix (and mean) is estimated...
        cov = EmpiricalCovariance().fit(x)

        # Mahalanobis distance (given the above covariance) from every sample to the mean (a measure of *outlierness*)
        distances = cov.mahalanobis(x)

        # the distances for the windows of interest are kept as scores
        scores = distances[i_windows_of_interest]

        roc_auc: float = roc_auc_score(labels, scores)
        average_precision: float = average_precision_score(labels, scores)

        precision, recall, thr = precision_recall_curve(labels, scores)
        mask = (precision >= min_precision)
        best_recall_at_min_precision: float = recall[mask].max().item()

        fpr, tpr, thresholds = roc_curve(labels, scores)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'<b>{algorithm_name}</b> (AUC={auc(fpr, tpr):.2f}, AP={average_precision:.2f}, BR@{min_precision}={best_recall_at_min_precision:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=500, height=500,
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        # fig.update_yaxes(scaleanchor="x", scaleratio=1)
        # fig.update_xaxes(constrain='domain')
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

        # return MahalanobisBasedMetrics(average_precision, best_recall_at_min_precision, roc_curve)
        return MahalanobisBasedMetrics(average_precision, best_recall_at_min_precision, fig)
    return (mahalanobis_based_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The *label* is `1` if `Event` is `True` and `0` otherwise.""")
    return


@app.cell
def _(no_undecided_res_df):
    y = no_undecided_res_df['Event'].astype(int)
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For VAE""")
    return


@app.cell
def _(mahalanobis_based_metrics, no_undecided_res_df, y, z_mean):
    res_vae = mahalanobis_based_metrics(z_mean, y, no_undecided_res_df.index.values, algorithm_name='VAE')
    res_vae[-1]
    return


@app.cell
def _(no_undecided_res_df, trapallada, y, z_mean):
    trapallada.results.analysis.mahalanobis_based_metrics(z_mean, y, no_undecided_res_df.index.values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For AR""")
    return


@app.cell
def _(ar_df, mahalanobis_based_metrics, no_undecided_res_df, y):
    res_ar = mahalanobis_based_metrics(ar_df.filter(regex='z_\\d+').values, y, no_undecided_res_df.index.values, algorithm_name='AR')
    res_ar[-1]
    return


if __name__ == "__main__":
    app.run()
