import marimo

__generated_with = "0.14.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"> Annotation tool")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pathlib
    from datetime import datetime
    from io import BytesIO

    import yaml
    import numpy as np
    import pandas as pd

    from torch.utils.data import DataLoader

    import plotly

    import trapallada.analysis
    import trapallada.results.analysis
    import trapallada.results.plot
    import trapallada.model.ar
    return BytesIO, pathlib, pd, trapallada, yaml


@app.cell
def _():
    # so that *plotly*'s pictures are correctly rendered
    import plotly.io as pio
    pio.renderers.default = 'notebook'
    return


@app.cell
def _():
    # today = datetime.now().strftime("%Y_%m_%d")
    return


@app.cell
def _():
    event_options = ['Interesting', 'NOT interesting', 'Undecided']
    # server_results_filename = f'res_{today}.xlsx'
    # host_results_filename = f'signals_{today}.xlsx'
    server_results_filename = f'res.xlsx'
    host_results_filename = f'signals.xlsx'
    return event_options, host_results_filename, server_results_filename


@app.cell
def _(event_options):
    event_mapping = dict(zip(event_options, event_options))
    event_mapping[event_options[-1]] = None
    return (event_mapping,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Instructions
    
        - Check the signals in the table one by one and select the appropriate label (and, maybe, fill in the *kind* of event)
        - When finished, click the `Download` button at the end to get an Excel file with the results.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"# Signals to assess")
    return


@app.cell
def _(mo):
    mo.md(r"")
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
def _(excel_file, server_results_filename):
    server_results_file= excel_file.parent / server_results_filename
    # server_results_file
    return (server_results_file,)


@app.cell
def _(mo):
    mo.md(r"Click on a signal to visualize it below.")
    return


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
def _(event_options, mo):
    mo.md(f"""Label the signal. If `{event_options[0]}` is selected, you can also write the *kind* of event (the content will be saved after clicking somewhere else)""")
    return


@app.cell
def _(event_radio, mo):
    if event_radio.value == 'Interesting':
        type_text = mo.ui.text(placeholder='')
        subtype_text = mo.ui.text(placeholder='')
        mo.output.append(
            mo.hstack((mo.vstack((mo.plain_text('Type'), type_text)), mo.vstack((mo.plain_text('Subtype'), subtype_text))))
        )
    return subtype_text, type_text


@app.cell
def _(event_mapping, full_signals_dir, mo, signals_table, trapallada):
    mo.stop(len(signals_table.value) == 0, mo.md("**Choose a signal**"))
    event_radio = mo.ui.radio(options=event_mapping, value=None)

    selected_row = signals_table.value.iloc[0]
    signal_id, *start_end = selected_row[['signal', 'start', 'end']]

    mo.vstack(
        (
            event_radio, 
            trapallada.results.plot.window(signal_id, full_signals_dir, *start_end))
    )
    return event_radio, selected_row


@app.cell
def _(mo):
    clear_button = mo.ui.run_button(label="Clear annotations")
    clear_button
    return (clear_button,)


@app.cell
def _(clear_button, excel_df, pd, server_results_file):
    # output `DataFrame` must be created only once (hence, it must be in a cell not re-run when it's modified)
    # labels_df = pd.DataFrame({'Event': None, 'Type': None}, index=excel_df.index)
    # labels_df = pd.DataFrame({'Type': None}, index=excel_df.index)
    if server_results_file.exists() and not clear_button.value:
        labels_df = pd.read_excel(server_results_file)
        labels_df = labels_df.drop(columns=labels_df.columns[0])
    else:
        labels_df = pd.DataFrame({'Label': None, 'Type': None, 'Subtype': None}, index=excel_df.index)
    # if clear_button.value:

    #     # no need to do anything since when the (whole) cell is re-run, `labels_df` is reset
    #     pass
    return (labels_df,)


@app.cell
def _(mo):
    mo.md(r"The annotations made are shown below, along with a button to download an Excel file with them.")
    return


@app.cell
def _(
    BytesIO,
    event_options,
    event_radio,
    excel_file,
    host_results_filename,
    labels_df,
    mo,
    selected_row,
    server_results_file,
    subtype_text,
    type_text,
):
    # if something was actually selected (if not, this cell is still run when a signal is selected, 
    # and `event_radio.value` is `None`, which could potentially destroy previously restored data)
    if event_radio.value is not None:

        # `DataFrame` is updated when the radio button is updated
        labels_df.loc[selected_row.name.item(),'Label'] = event_radio.value

        # if the first event (interesting) was selected...
        if event_radio.value == event_options[0]:
            labels_df.loc[selected_row.name.item(),'Type'] = type_text.value
            labels_df.loc[selected_row.name.item(),'Subtype'] = subtype_text.value

    # results are written to disk *in the server*
    labels_df.to_excel(excel_file.parent / server_results_file)

    # a buffer to hold in memory...
    buf = BytesIO()

    # ..the Excel file i
    labels_df.to_excel(buf)
    # labels_df.to_excel(buf, index=False)

    # rewind
    buf.seek(0)

    download_button = mo.download(data=buf, filename=host_results_filename)

    # both `DataFrame` and download button
    mo.vstack((labels_df, download_button))
    return


if __name__ == "__main__":
    app.run()
