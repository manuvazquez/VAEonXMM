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

    import yaml
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
    return pathlib, pd, trapallada, yaml


@app.cell
def _():
    import plotly.io as pio
    pio.renderers.default = 'notebook'
    return


@app.cell
def _(mo):
    event_options = ["Interesting", "NOT interesting", "Undecided"]
    event_radio = mo.ui.radio(options=event_options, value=event_options[-1])
    event_text = mo.ui.text(placeholder='')
    return event_radio, event_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Input Files""")
    return


@app.cell
def _():
    # excel_file_browser = mo.ui.file_browser(
    #     initial_path='results', filetypes=['.xlsx'],selection_mode='file', multiple=False, restrict_navigation=True
    # )
    # excel_file_browser
    return


@app.cell
def _(mo, pd, trapallada):
    # excel_file = excel_file_browser.path()
    excel_file = trapallada.util.get_base_directory() / 'results' / 'multichannel' / 'rates_large_infs' / 'time_z_5' / 'signals_to_assess.xlsx'
    assert excel_file.exists()
    mo.stop(excel_file is None, mo.md("**Choose an Excel file**"))
    excel_df = pd.read_excel(excel_file)
    return excel_df, excel_file


@app.cell
def _(excel_df, mo):
    signals_table = mo.ui.table(excel_df, selection='single')
    signals_table
    return (signals_table,)


@app.cell
def _(excel_file, pathlib, yaml):
    parameters_file = excel_file.parent / 'extended_parameters.yaml'
    assert parameters_file.exists()
    with parameters_file.open() as f:
        parameters = yaml.safe_load(f)
    full_signals_dir = (pathlib.Path.home() / parameters['dataset']['files']['signals_file']).parent / 'full_signals'
    assert full_signals_dir.exists()
    return (full_signals_dir,)


@app.cell
def _(event_radio, event_text, mo):
    if event_radio.value == 'Interesting':
        mo.output.append(event_text)
    return


@app.cell
def _(event_radio, full_signals_dir, mo, signals_table, trapallada):
    mo.stop(len(signals_table.value) == 0, mo.md("**Choose a signal**"))
    selected_row = signals_table.value.iloc[0]
    signal_id, *start_end = selected_row[['signal', 'start', 'end']]
    # mo.output.append(trapallada.results.plot.window(signal_id, full_signals_dir, *start_end))
    mo.vstack(
        (
            event_radio, 
            trapallada.results.plot.window(signal_id, full_signals_dir, *start_end))
    )
    return (selected_row,)


@app.cell
def _(excel_df, pd):
    labels_df = pd.DataFrame({'Event': None, 'Type': None}, index=excel_df.index)
    return (labels_df,)


@app.cell
def _(event_radio, event_text, labels_df, selected_row):
    labels_df.loc[selected_row.name.item(),'Label'] = event_radio.value
    labels_df.loc[selected_row.name.item(),'Type'] = event_text.value
    labels_df.to_excel('res.xlsx')
    labels_df
    return


if __name__ == "__main__":
    app.run()
