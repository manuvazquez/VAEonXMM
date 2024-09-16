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
import yaml

import torch

from jaxtyping import Array, Float

from astropy.table import Table
import astropy.io.fits as fits
from astropy.time import Time
import astropy.units as u

import trapallada.util
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

    # file is read with the high-level interface (`unit_parse_strict="silent"` is used to avoid warnings in an irrelevant column)
    df = Table.read(file, format='fits', hdu=1, unit_parse_strict="silent").to_pandas()

    # the "Mission Time Zero" is extracted from the header...
    mjdref = fits.getheader(file, 1)['MJDREF']
    
    # ...and an *astropy* `Time` object is made out of it
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

def get_fields_from_header(file: str | pathlib.Path, fields: Sequence[str]) -> tuple:
    
    header = fits.getheader(file, 1)

    return [header[e] for e in fields]

def get_position_from_header(file: str | pathlib.Path) -> pd.Series:

    keys = ['SRC_RA', 'SRC_CRA', 'SRC_DEC', 'SRC_CDEC']

    values = get_fields_from_header(file, keys)

    return pd.Series(dict(zip([e.replace("SRC_", "") for e in keys], values)))

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
        ) -> Float[Array, 'window time'] | tuple[Float[Array, 'window time'], int]:
    """
    Vertically slices a DataFrame into windows of a specified size and step.
    
    Args:
        df (pd.DataFrame): The DataFrame to be sliced.
        window_size (int): The size of each window.
        step (int): The step size between windows.
        tail_as_window (bool, optional): Whether to include a last window with leftovers. Defaults to True.
        return_n_windows (bool, optional): Whether to return the number of windows. Defaults to False.
    
    Returns:
        Tuple[Float[Array, 'window time']]: A tuple containing the stacked slices and the corresponding index slices.
        If `return_n_windows` is True, the tuple also includes the number of windows.
    """
    
    # data is sliced...
    slices = np.lib.stride_tricks.sliding_window_view(df.values, window_shape=window_size, axis=0)[::step]

    # ...and so are the time indexes
    index_slices = np.lib.stride_tricks.sliding_window_view(df.index.values, window_shape=window_size)[::step]

    # the 1st window for the first signal, then the 1st for the 2nd signal...
    # then the 2nd window for the 1st signal, and so forth and so on
    stacked_slices = slices.reshape((-1, slices.shape[-1]))

    # the number of resulting windows
    n_windows = slices.shape[0]

    # if a last window with (most likely including leftovers) is to be considered...
    if tail_as_window:

        stacked_slices = np.vstack((stacked_slices, df.values[-window_size:].T))

        index_slices = np.vstack((index_slices, df.index.values[-window_size:][None, :]))

        n_windows += 1

    index_slices = np.repeat(index_slices, len(df.columns), axis=0)

    res = stacked_slices, index_slices

    # if the number of windows/slices is to be returned...
    if return_n_windows:

        # ...it is appended to the (`tuple`) result
        res = res + (n_windows,)

    return res

def tars_to_array(
        files: Sequence[str | pathlib.Path], bin_size: int = 63, window_size: int = 200, step: int = 50,
        max_n_sources: int = 4, min_power: float = 0.) -> tuple[Float[Array, 'window time'], pd.DataFrame]:
    """
    Converts a list of tar files into a NumPy array of signals and a pandas DataFrame of metadata.
    
    Args:
        files (Sequence[str | pathlib.Path]): A sequence of file paths to tar files.
        bin_size (int, optional): The number of measurements that make up a bin. Defaults to 63.
        window_size (int, optional): The size (in bins) of the window for signal slicing. Defaults to 200.
        step (int, optional): The step size (in bins) for signal slicing. Defaults to 50.
        max_n_sources (int, optional): The maximum number of sources to read from each tar file. Defaults to 4.
        min_power (float, optional): The minimum power of a source to be kept. Defaults to 0 (i.e., no signal is dismissed on account of low power).
    
    Returns:
        tuple[Float[Array, 'window time'], pd.DataFrame]: A tuple containing the NumPy array of signals and the pandas DataFrame of metadata.
    
    Raises:
        RuntimeError: If the period of a tar file does not match the previous period(s).
    """

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

            # if there is *any* `NaN` at all (in any column, at any time instant)...
            if df.isna().any().any():

                # logger.warning(f'"{f}" has NaN values: interpolating...')
                
                # ...it's filled in using interpolation.
                df = df.interpolate(method='linear', axis=0)

                if df.isna().any().any():

                    logger.warning(f'"{f}" still has `NaN` values after interpolating, skipping...')
                    
                    continue

            # the period of the raw `DataFrame`
            period = df.index[1] - df.index[0]
            
            # signals are binned
            binned_df = trapallada.xsa.util.bin_fits_df(df, bin_size=bin_size)

            # ---

            # min_power = 5.
            
            # power of the (binned) signals
            power = (binned_df**2).mean(axis=0)

            # a `Series` with the columns (i.e., signals) whose power is below the threshold
            low_power_columns = power[(power < min_power)].index
            n_low_power_columns = len(low_power_columns)

            # if every signal has low power...
            if n_low_power_columns == len(binned_df.columns):

                logger.warning(f'every signal has power below {min_power} in "{f}", skipping...')
                
                # ...the file is skipped
                continue

            # at this point not EVERY signal has low power, but those that do have...
            if n_low_power_columns > 0:

                # ...are dropped
                binned_df = binned_df.drop(low_power_columns, axis=1)

                logger.warning(f'dropping {n_low_power_columns} signals in "{f}" due to low power (<{min_power})')

            # ---

            # the period of the *binned* `DataFrame`
            binned_period = binned_df.index[1] - binned_df.index[0]

            # if this is not the first file (`last_period` is `None`)...
            if last_period:

                if period != last_period:

                    logger.critical(f'the period of "{f}" does not match the previous period(s)')
                    
                    raise RuntimeError('Period mismatch')

            # if this IS the first file...
            else:

                window_size_in_seconds = (binned_period * window_size).total_seconds()

                # print(step)
                logger.info(f'Requested a window size of, effectively, duration {window_size_in_seconds} seconds '
                            f'(period is {binned_period.total_seconds()} seconds, step is {(binned_period * step).total_seconds()} seconds)')

            last_period = period

            if len(binned_df) < window_size:

                logger.warning(f'skipping "{f}" because it is too short after binning')
                
                continue

            # signals are "sliced"
            sliced_signals, sliced_times, n_windows = trapallada.xsa.util.slice(
                binned_df, window_size=window_size, step=step, tail_as_window=True, return_n_windows=True)

            signals_list.append(sliced_signals)
            start_list.append(sliced_times[:, 0])
            end_list.append(sliced_times[:, -1])
            f_list.append(np.full(len(sliced_signals), f.name))
            names_list.append(np.tile(binned_df.columns, n_windows))

    return np.vstack(signals_list), pd.DataFrame(
        {'start': np.hstack(start_list), 'end': np.hstack(end_list), 'tar file': np.hstack(f_list), 'source': np.hstack(names_list)})


def cl_tars_to_array(args: list = None):
    """
    Command line interface for the `tars_to_array` function.
    """

    parser = argparse.ArgumentParser(description='Download observations.')

    parser.add_argument('download_directory', type=pathlib.Path, help='Download directory')
    parser.add_argument('output_directory', type=pathlib.Path, help='Output directory')
    parser.add_argument('output_files_stem', type=str, help='Output directory')
    
    parser.add_argument('-b', '--bin-size', type=int, default=None, help='Bin size')
    parser.add_argument('-w', '--window-size', type=int, default=None, help='Window size (in bins)')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step (in bins) for the sliding window')
    parser.add_argument('-M', '--max-n-sources', type=int, default=None, help='Maximum number of sources')
    parser.add_argument('-m', '--min-power', type=float, default=None, help='Minimum power')

    command_line_arguments = parser.parse_args(args)
    
    # tar_files = list(command_line_arguments.download_directory.glob('*.tar'))[:4]
    tar_files = list(command_line_arguments.download_directory.glob('*.tar'))

    # arguments that are `None`, and also command line-only arguments are trimmed
    not_none_args = {
        k: v for k, v in vars(command_line_arguments).items()
        if (v is not None) and (k not in ['download_directory', 'output_directory', 'output_files_stem'])
    }

    signals, metadata_df = tars_to_array(tar_files, **not_none_args)

    duration = (metadata_df['end'] - metadata_df['start']).iloc[0].total_seconds()

    logger.info(f'{signals.shape[0]} signals of length {signals.shape[1]} samples ({duration} seconds)')

    torch_file = (command_line_arguments.output_directory / (command_line_arguments.output_files_stem + '_signals')).with_suffix('.pt')
    parquet_file = (command_line_arguments.output_directory / (command_line_arguments.output_files_stem + '_metadata')).with_suffix('.parquet')
    yaml_file = (command_line_arguments.output_directory / (command_line_arguments.output_files_stem + '_hyperparameters')).with_suffix('.yaml')

    # torch.save(torch.from_numpy(np.expand_dims(signals, 1)), torch_file)
    torch.save(np.expand_dims(signals, 1), torch_file) # saved as numpy array
    metadata_df.to_parquet(parquet_file)

    # the default arguments of the function doing the actual work...
    default_args = trapallada.util.func_args_with_defaults(trapallada.xsa.util.tars_to_array)
    
    # ...minus the `files` argument
    default_args.pop('files')
    
    with open(yaml_file, 'w') as f:
        
        # `default_args` are used only whenever a key is not present in `not_none_args`
        yaml.dump(default_args | not_none_args, f, default_flow_style=False)