import pathlib
import time
import os
import argparse
import tarfile
import tempfile
import shutil
import re
from typing import Sequence

import pandas as pd

import numpy as np

from jaxtyping import Array, Float

from astropy.table import Table
import astropy.io.fits as fits
from astropy.time import Time
import astropy.units as u

import trapallada.xsa.access
import trapallada.logging as log

logger = log.setup_logging(__name__)

# SRC: Source
# TSR: Type of data and processing stage
# file_types = ['SRCTSR', 'SRSPEC', 'OBSMLI', 'OMSRLI', 'SRCRES']
file_types = ['SRCTSR']

# PN: Epic-pn camera
# re_filename = re.compile(r'^P(\d{10})PN([SX]\d{3})(' + '|'.join(file_types) + r')([0-9A-F]+)\.\w+$')
re_filename = re.compile(r'^P(\d{10})PN(S\d{3})(' + '|'.join(file_types) + r')([0-9A-F]+)\.\w+$')

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
    first_file = None
    
    for f in files:

        # print(f'Reading {f}...')
        
        df = read_fits_file(f)
        
        if 'TIME' in df:

            series = df['RATE']
            series.index = df['TIME']

            # if this is the first file...
            if index is None:
                
                # ...the reference index...
                index = series.index

                # ...and file are set
                first_file = f
            
            # if this is NOT the first file
            else:

                if not index.equals(series.index):

                    logger.warning((f'Time stamps for file "{f.relative_to(f.parents[1])}" don\'t match '
                            f'those for "{first_file.relative_to(first_file.parents[1])}" (the first file, taken as reference), skipping...'))
                    
                    skipped_files_list.append(f)
                    
                    continue

                # assert index.equals(series.index), 'The indexes of the time series do not match.'
            
            series.name = f.name
            series_list.append(series)
            
        # if `TIME` column is NOT present...
        else:
            
            logger.warning(f'{f.relative_to(f.parents[1])} does not look like a light curve, skipping...')
            
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
    fits_files_df = fits_files_series.str.extract(trapallada.xsa.util.re_filename)

    # ...which is then enriched with the full path of every file
    fits_files_df['full path'] = fits_files
    
    return fits_files_df.sort_values(3)

def download_observations(
        download_directory: str | pathlib.Path,
        duration_interval: Sequence[int] = (35_000, 45_000),
        sleep_time_between_requests: float = 5.) -> None:

    # metadata for the observations in the given interval
    obs_metadata = trapallada.xsa.access.get_observations_in_interval(duration_interval=duration_interval)

    # their id's are extracted
    obs_to_download = obs_metadata['observation_id']

    # for every observation...
    for obs_id in obs_to_download:

        # print(f'Requesting observation {obs_id}')
        logger.info(f'Requesting observation {obs_id}')
        
        # ...the data is downloaded
        tar_file = trapallada.xsa.access.download_obs(obs=obs_id, output_directory=download_directory)

        # if the file has been created less than a minute ago (i.e., something was *actually* downloaded)...
        if (tar_file is not None) and (time.time() - os.path.getmtime(tar_file)) < 60:
        
            # in order to, hopefully, avoid bans/errors
            time.sleep(sleep_time_between_requests)

def cl_download_observations(args: list = None):
    """
    Command line interface for the `download_observations` function.
    """

    parser = argparse.ArgumentParser(description='Download observations.')

    parser.add_argument('download_directory', type=str, help='Download directory')
    
    parser.add_argument('-m', '--min-duration', type=int, default=35000, help='Minimum duration')
    parser.add_argument('-M', '--max-duration', type=int, default=45000, help='Maximum duration')

    command_line_arguments = parser.parse_args(args)

    download_directory = pathlib.Path(command_line_arguments.download_directory)
    
    if not download_directory.exists():
        print(f'"{download_directory}" does not exist')
        return

    # print(command_line_arguments)

    download_observations(
        download_directory=download_directory,
        duration_interval=(command_line_arguments.min_duration, command_line_arguments.max_duration))

def extract_filter_tar(tar_file: pathlib.Path | str, output_directory: pathlib.Path | str | None = None) -> pathlib.Path | None:
    """
    Extracts tha *tar* file for a given observation keeping only the files whose name match a specific pattern.

    Args:
        tar_file (pathlib.Path | str): Name of the *tar* file.
        output_directory (pathlib.Path | str): The output directory where the files will be saved.

    Returns:
        pathlib.Path or None: The path to the output directory where the files are saved (even if nothing was actually extracted)
        or `None` if there are not relevant files.
    """

    # for the sake of readability
    tar_file = pathlib.Path(tar_file)

    # if no output directory is given...
    if output_directory is None:
            
        # ...the parent directory of the *tar* file is used
        output_directory = tar_file.parent

    # the subdirectory is named after the *tar* file
    output_subdir = pathlib.Path(output_directory) / tar_file.stem

    # if it has been downloaded/extracted before...
    if output_subdir.exists():

        print(f'found directory "{output_subdir}", skipping...')
        
    # if it has NOT been downloaded/extracted before...
    else:

        # directory is created
        output_subdir.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:

            # context's descriptor is turned into a `Pathlib`
            tmp_dir_root = pathlib.Path(tmp_dir)

            tmp_dir_extracted = tmp_dir_root / 'extracted'
            
            with tarfile.open(tar_file) as f:

                try:

                    files = f.getnames()

                except tarfile.ReadError:

                    logger.error(f'There is something wrong with downloaded tar file "{tar_file}"')

                    raise
                
                f.extractall(path=tmp_dir_extracted)
            
            # only file (names) matching the pattern are kept...
            m_light_curves = [(f, m) for f in files if (m := re_filename.match(pathlib.Path(f).name))]

            if not m_light_curves:
                    
                    logger.warning(f'No relevant files in "{tar_file}"')
    
                    return None
            
            # ...and every one of the them is copied into the appropriate directory
            for filename, _ in m_light_curves:
                
                shutil.copy(tmp_dir_extracted / filename, output_subdir)

    return output_subdir

def download_extract_filter(
        obs: str, output_directory: pathlib.Path | str, keep_tar: bool = False, n_attemps: int = 3, verbose: bool = True) -> pathlib.Path:
    """
    Downloads and extracts the 'tar' file for a given observation keeping only the files whose name match a specific pattern.
    Data already processed (through the Pipeline Processing System, or PPS) is returned.

    Args:
        obs (str): The observation identifier.
        output_directory (pathlib.Path | str): The output directory where the files will be saved.
        keep_tar (bool, optional): Whether to keep the downloaded tar file. Defaults to False.
        n_attemps (int, optional): Number of attemps to download the file. Defaults to 3.
        verbose (bool, optional): Whether to print download messages. Defaults to True.

    Returns:
        pathlib.Path: The path to the output directory where the files are saved (even if nothing was actually downloaded).
    """

    # for the sake of convenience
    tar_file = pathlib.Path(output_directory) / f'{obs}.tar'

    if tar_file.exists():
        
        logger.info(f'found file "{tar_file}", extracting...')

    else:
        
        tar_file = trapallada.xsa.access.download_obs(obs=obs, output_directory=output_directory, n_attemps=n_attemps, verbose=verbose)

    output_subdir = extract_filter_tar(tar_file)

    return output_subdir

def read_tar(file: str | pathlib.Path, max_n_sources: int = 3) -> pd.DataFrame | None:
    """
    Reads a tar file containing FITS files, returning a number of sources.

    Args:
        file (str or pathlib.Path): The path to the tar file.
        max_n_sources (int, optional): The maximum number of top ranked files to read. Defaults to 3.

    Returns:
        pd.DataFrame or None: The DataFrame containing the the top *ranked* FITS files or `None` if there are not relevant files.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        # context's descriptor is turned into a `Pathlib`
        tmp_dir_root = pathlib.Path(tmp_dir)
        
        # files are extracted into a temporary directory...
        extracted_files_dir = trapallada.xsa.util.extract_filter_tar(file, tmp_dir_root)

        # if no relevant files were found
        if extracted_files_dir is None:
                
                return None

        # ...and ranked
        ranked_files_df = trapallada.xsa.util.rank_obs_files(extracted_files_dir).iloc[:max_n_sources]['full path']

        # the top ranked files are read into a DataFrame
        fits_df, skipped_files_list = trapallada.xsa.util.read_fits_files(ranked_files_df.tolist())

    return fits_df

def slice(
        df: pd.DataFrame, window_size: int, step: int, tail_as_window: bool = True, return_n_windows: bool = False
        ) -> Float[Array, 'window time']:
    
    slices = np.lib.stride_tricks.sliding_window_view(df.values, window_shape=window_size, axis=0)[::step]

    index_slices = np.lib.stride_tricks.sliding_window_view(df.index.values, window_shape=window_size)[::step]

    # the 1st window for the first signal, then the 1st for the 2nd signal...
    # then the 2nd window for the 1st signal, and so forth and so on
    stacked_slices = slices.reshape((-1, slices.shape[-1]))

    n_windows = slices.shape[0]

    if tail_as_window:

        stacked_slices = np.vstack((stacked_slices, df.values[-window_size:].T))

        index_slices = np.vstack((index_slices, df.index.values[-window_size:][None, :]))

        n_windows += 1

    index_slices = np.repeat(index_slices, len(df.columns), axis=0)

    res = stacked_slices, index_slices

    if return_n_windows:

        res = res + (n_windows,)

    return res

def tars_to_array(
        files, bin_size: int = 63, window_size: int = 200, step: int = 50, min_duration: int = 20_000, max_n_sources: int = 4
        ) -> tuple[Float[Array, 'window time'], pd.DataFrame]:

    signals_list = []
    last_period = None

    start_list, end_list = [], []
    f_list = []
    names_list = []

    # every tar file...
    for f in files:
        
        # ...is read into a `DataFrame`
        df = trapallada.xsa.util.read_tar(f, max_n_sources=max_n_sources)

        # if something was actually read...
        if df is not None:

            # duration of the signal in this file
            duration = df.index[-1] - df.index[0]

            if duration.total_seconds() < min_duration:
                
                logger.warning(f'skipping "{f}" because it is too short')
                
                continue

            # the period
            period = df.index[1] - df.index[0]

            # if this is not the first file...
            if last_period:

                assert period == last_period, f'the period of "{f}" does not match the previous period'

            # if there is *any* `NaN` at all (in any column, at any time instant)...
            if df.isna().any().any():
                
                # ...it's filled in using interpolation.
                df = df.interpolate(method='linear', axis=0)

            # signals are binned
            binned_df = trapallada.xsa.util.bin_fits_df(df, bin_size=bin_size)

            sliced_signals, sliced_times, n_windows = trapallada.xsa.util.slice(
                binned_df, window_size=window_size, step=step, tail_as_window=True, return_n_windows=True)

            signals_list.append(sliced_signals)
            start_list.append(sliced_times[:, 0])
            end_list.append(sliced_times[:, -1])
            f_list.append(np.full(len(sliced_signals), f.name))
            names_list.append(np.tile(binned_df.columns, n_windows))

    return np.vstack(signals_list), pd.DataFrame(
        {'start': np.hstack(start_list), 'end': np.hstack(end_list), 'tar file': np.hstack(f_list), 'source': np.hstack(names_list)})