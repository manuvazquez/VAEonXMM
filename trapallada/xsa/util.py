import pathlib

import pandas as pd

from astropy.table import Table
import astropy.io.fits as fits
from astropy.time import Time
import astropy.units as u

import trapallada.xsa.access

# # in order to avoid warnings, custom unit 'fraction' is defined
# fraction = u.def_unit('fraction')

def bin_fits_df(df: pd.DataFrame, bin_size: int = 63) -> pd.DataFrame:
    """
    Bin all the columns in a FITS DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        bin_size (int, optional): The size of the rolling window. Defaults to 63.

    Returns:
        pd.DataFrame: The measurements.
    """

    # one could also exploit the dates...but then no `step` is allowed
    # ts = df.set_index('date')['RATE'].rolling(window=pd.Timedelta(100, unit='s')).sum()
    
    # a *rolling window* is used to bin-collapse together different measurements;
    # the *label* (within the index) of every window is that of the last element in the window;
    # the very first element is dismissed because there is not enough elememts (going backwards) to fill the window
    return df.rolling(window=bin_size, center=False, step=bin_size).sum().iloc[1:]


def read_fits_file(file: pathlib.Path | str) -> pd.DataFrame:
    """
    Read a FITS file and return its content as a DataFrame.

    Args:
        file (pathlib.Path | str): The file to be read.

    Returns:
        pd.DataFrame: The content of the FITS file.
    """

    # # file is read with the high-level interface...
    # table = Table.read(file, format='fits', hdu=1)

    # # ...and turned into a Pandas DataFrame
    # return table.to_pandas()

    # file is read with the high-level interface (`unit_parse_strict="silent"` is used to avoid warnings in an irrelevant column)
    df = Table.read(file, format='fits', hdu=1, unit_parse_strict="silent").to_pandas()

    # the "Mission Time Zero" is extracted from the header...
    mjdref = fits.getheader(file, 1)['MJDREF']
    
    # and an *astropy* `Time` object is made out of it
    time_ref = Time(mjdref, format='mjd')
    
    # dates are shifted
    time_dates = time_ref + df['TIME'].values * u.s
    
    # df.index = pd.to_datetime(time_dates.to_datetime())
    df['TIME'] = pd.to_datetime(time_dates.to_datetime())

    return df

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
            
            series = df['RATE']
            series.index = df['TIME']
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

def rank_obs_files(obs_dir: str | pathlib.Path) -> pd.DataFrame:
    """
    Ranks the observed sources in the given directory.

    Args:
        obs_dir (str or pathlib.Path): The directory containing the FITS files.

    Returns:
        pd.DataFrame: A DataFrame containing the ranked observed sources.
    """
    # all the FITS files in the given directory
    fits_files = list(obs_dir.glob('*.FTZ'))
    
    # a Series with just the names of every file
    fits_files_series = pd.Series([f.name for f in fits_files], dtype=pd.StringDtype())
    
    # the names are parsed into a `DataFrame`...
    fits_files_df = fits_files_series.str.extract(trapallada.xsa.access.re_filename)

    # ...which is then enriched with the full path of every file
    fits_files_df['full path'] = fits_files
    
    return fits_files_df.sort_values(3)