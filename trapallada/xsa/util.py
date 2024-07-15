import pathlib

import pandas as pd
from astropy.table import Table

def time_series_from_df(df: pd.DataFrame, window_size: int = 63) -> pd.DataFrame:
    """
    Extract a time series from the a FITS DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        window_size (int, optional): The size of the rolling window. Defaults to 63.

    Returns:
        pd.DataFrame: The measurements.
    """

    # one could also exploit the dates...but then no `step` is allowed
    # ts = df.set_index('date')['RATE'].rolling(window=pd.Timedelta(100, unit='s')).sum()
    
    # a *rolling window* is used to bin-collapse together different measurements;
    # the *label* (within the index) of every window is that of the last element in the window;
    # the very first element is dismissed because there is not enough elememts (going backwards) to fill the window
    return df.set_index('TIME')['RATE'].rolling(window=window_size, center=False, step=window_size).sum().iloc[1:]

def read_fits_file(file: pathlib.Path | str) -> pd.DataFrame:
    """
    Read a FITS file and return its content as a DataFrame.

    Args:
        file (pathlib.Path | str): The file to be read.

    Returns:
        pd.DataFrame: The content of the FITS file.
    """

    # file is read with the high-level interface...
    table = Table.read(file, format='fits', hdu=1)
    
    # ...and turned into a Pandas DataFrame
    return table.to_pandas()

    # # 'TIME' column is converted to datetime
    # df['date'] = pd.to_datetime(df['TIME'], unit='s')

    # return df


def read_fits_files(files: list[pathlib.Path | str]) -> tuple[pd.DataFrame, list[pathlib.Path | str]]:
    """
    Read a list of FITS files and return their content as a single DataFrame.

    Args:
        files (list[pathlib.Path | str]): The files to be read.

    Returns:
        pd.DataFrame: The content of the FITS files.
        list[pathlib.Path | str]: The list of files that have been skipped.
    """

    series_list = []
    skipped_files_list = []
    index = None
    
    for f in files:
        
        df = read_fits_file(f)
        
        if 'TIME' in df:
            series = time_series_from_df(df)
            series.name = f.name
            series_list.append(series)
            
            # if this is the first file
            if index is None:
                index = series.index
            # if this is NOT the first file
            else:
                assert index.equals(series.index), 'The indexes of the time series do not match.'
        else:
            # print(f'{f} does not look like a light curve, skipping...')
            skipped_files_list.append(f)

    return pd.concat(series_list, axis=1), skipped_files_list