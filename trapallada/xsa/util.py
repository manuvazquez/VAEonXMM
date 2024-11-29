import pathlib
import time
import os
import argparse
import tarfile
import tempfile
import shutil
import re
import pickle
from typing import Sequence

import pandas as pd
import numpy as np
import yaml

import torch

from jaxtyping import Array, Float, Bool

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

def read_fits_file(file: pathlib.Path | str, return_gti: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a FITS file and return its content as a DataFrame.

    Args:
        file (pathlib.Path | str): The file to be read.

    Returns:
        tuple: A DataFrame with the signals in the FITS file and a mask with the good time intervals.
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

    # "good time intervals" are read, times adjusted...
    gti_time_dates = time_ref +\
            Table.read(file, format='fits', hdu=2, unit_parse_strict="silent").to_pandas().values * u.s
    
    # ...turned into Pandas `datetime`s, and (re)packed as a `DataFrame`
    gti_df = pd.DataFrame(
        pd.to_datetime(
            gti_time_dates.to_datetime().ravel()
        ).values.reshape((-1, 2)),
        columns=['start', 'end']
    )
    
    mask = pd.Series([False] * len(df), index=df['TIME'])
    for _, row in gti_df.iterrows():
        mask[(mask.index > row['start']) & (mask.index < row['end'])] = True

    if return_gti:
        return df, mask, gti_df
    else:
        return df, mask

def read_fits_files(
        # files: list[pathlib.Path | str], columns_of_interest: list[str] = ['RATE', 'BACKV']
        files: list[pathlib.Path | str], columns_of_interest: list[str] = ['RATE']
    ) -> tuple[pd.DataFrame, list[pathlib.Path | str], pd.Series]:
    """
    Read a list of FITS files and return their content as a single DataFrame.

    Args:
        files (list[pathlib.Path | str]): The files to be read.

    Returns:
        tuple[pd.DataFrame, list[pathlib.Path | str], pd.DataFrame]: 
            - The content of the FITS files.
            - The list of files that have been skipped.
            - A DataFrame containing the header information.
    """

    df_list = []
    mask_list = []
    skipped_files_list = []
    index = None
    first_file = None
    headers_list = []

    for f in files:

        df, mask_series = read_fits_file(f)
        
        if 'TIME' in df:

            cols_of_interest_df = df[columns_of_interest + ['TIME']].set_index('TIME')
            cols_of_interest_df.columns = pd.MultiIndex.from_tuples([(f.name, c) for c in cols_of_interest_df.columns])

            # if this is the first file...
            if index is None:
                
                # ...the reference index...
                index = df['TIME']

                # ...and file are set
                first_file = f
            
            # if this is NOT the first file
            else:

                if not index.equals(df['TIME']):

                    logger.warning((f'Time stamps for file "{f.relative_to(f.parents[1])}" don\'t match '
                            f'those for "{first_file.relative_to(first_file.parents[1])}" (the first file, taken as reference), skipping...'))
                    
                    skipped_files_list.append(f)
                    
                    continue

                # assert index.equals(series.index), 'The indexes of the time series do not match.'
            
            df_list.append(cols_of_interest_df)
            headers_list.append(get_position_from_header(f))

            mask_series.name = f.name
            mask_list.append(mask_series)
            
        # if `TIME` column is NOT present...
        else:
            
            logger.warning(f'{f.relative_to(f.parents[1])} does not look like a light curve, skipping...')
            
            skipped_files_list.append(f)

    signals_df = pd.concat(df_list, axis=1)
    signals_df.columns.names = ['file', 'data']

    header_df = pd.concat(headers_list, axis=1).T
    header_df.index = signals_df.columns.get_level_values('file').unique()

    mask_df = pd.concat(mask_list, axis=1)

    return signals_df, skipped_files_list, header_df, mask_df

def get_fields_from_header(file: str | pathlib.Path, fields: Sequence[str]) -> tuple:
    
    header = fits.getheader(file, 1)

    return [header[e] for e in fields]

def get_position_from_header(file: str | pathlib.Path) -> pd.Series:

    # keys = ['SRC_RA', 'SRC_CRA', 'SRC_DEC', 'SRC_CDEC']
    keys = ['SRC_RA', 'SRC_DEC', 'SRC_CRA', 'SRC_CDEC']

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

def read_tar(
        file: str | pathlib.Path, max_n_sources: int = 4, signals_to_extract: list[str] = ['RATE']
    ) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """
    Reads a tar file containing FITS files, returning a number of sources.

    Args:
        file (str or pathlib.Path): The path to the tar file.
        max_n_sources (int, optional): The maximum number of top ranked files to read. Defaults to 4.
        signals_to_extract (list[str], optional): The signals to be extracted. Defaults to ['RATE'].

    Returns:
        tuple[pd.DataFrame or None, pd.Series or None]: A tuple containing the DataFrame with the top ranked FITS files
        and a Series with the header information, or `None` if there are no relevant files.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        # context's descriptor is turned into a `Pathlib`
        tmp_dir_root = pathlib.Path(tmp_dir)
        
        # files are extracted into a temporary directory...
        extracted_files_dir = extract_filter_tar(file, tmp_dir_root)

        # if no relevant files were found
        if extracted_files_dir is None:
                
                return None, None, None

        # ...and ranked
        ranked_files_df = rank_obs_files(extracted_files_dir).iloc[:max_n_sources]['full path']

        # the top ranked files are read into a DataFrame
        fits_df, skipped_files_list, header_df, mask_df = read_fits_files(ranked_files_df.tolist(), columns_of_interest=signals_to_extract)

    return fits_df, header_df, mask_df

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
    
    # for the sake of convenience
    unsampled_slices: Float[Array, 'window signal/channel time'] = np.lib.stride_tricks.sliding_window_view(df.values, window_shape=window_size, axis=0)

    # data is sliced...
    # slices: Float[Array, 'window signal/channel time'] = np.lib.stride_tricks.sliding_window_view(df.values, window_shape=window_size, axis=0)[::step]
    slices: Float[Array, 'window signal/channel time'] = unsampled_slices[::step]

    # ...and so are the time indexes
    index_slices: Float[Array, 'window time'] = np.lib.stride_tricks.sliding_window_view(df.index.values, window_shape=window_size)[::step]

    n_columns_per_signal = len(df.columns.get_level_values('data').unique())

    # the 1st window for the first signal, then the 1st for the 2nd signal...
    # then the 2nd window for the 1st signal, and so forth and so on
    # stacked_slices = slices.reshape((-1, slices.shape[-1]))
    stacked_slices: Float[Array, 'window channel time'] = slices.reshape((-1, n_columns_per_signal, slices.shape[-1]))

    # the number of resulting windows
    n_windows = slices.shape[0]

    # if a last window with (most likely including leftovers) is to be considered...
    if tail_as_window:

        # if the entire signal is covered by the above windows...
        if (unsampled_slices.shape[0] % step) == 1:

            # ...there is no need to consider a last window
            
            logger.info(f'an integer number of windows in {df.columns.get_level_values(0).tolist()} (signals have length {len(df)}), no need to add "tail" windows...')

        # if the entire signal is NOT covered by the above windows...
        else:

            # stacked_slices = np.vstack((stacked_slices, df.values[-window_size:].T))
            stacked_slices = np.vstack((stacked_slices, df.values[-window_size:].T.reshape((-1, n_columns_per_signal, slices.shape[-1]))))

            index_slices = np.vstack((index_slices, df.index.values[-window_size:][None, :]))

            n_windows += 1

    # index_slices = np.repeat(index_slices, len(df.columns), axis=0)
    index_slices = np.repeat(index_slices, len(df.columns) // n_columns_per_signal, axis=0)

    res = stacked_slices, index_slices

    # if the number of windows/slices is to be returned...
    if return_n_windows:

        # ...it is appended to the (`tuple`) result
        res = res + (n_windows,)

    return res

def tars_to_array(
        files: Sequence[str | pathlib.Path], bin_size: int = 63, window_size: int = 200, step: int = 50,
        max_n_sources: int = 4, min_power: float = 0., min_count_rate: float = 0., min_gti_ratio: float = 0.,
        signals_to_extract: list[str] = ['RATE'],
        output_directory: str | pathlib.Path | None = None
        ) -> tuple[Float[Array, 'window channel time'], pd.DataFrame, pd.DataFrame]:
    """
    Converts a list of tar files into a NumPy array of signals and a pandas DataFrame of metadata.
    
    Args:
        files (Sequence[str | pathlib.Path]): A sequence of file paths to tar files.
        bin_size (int, optional): The number of measurements that make up a bin. Defaults to 63.
        window_size (int, optional): The size (in bins) of the window for signal slicing. Defaults to 200.
        step (int, optional): The step size (in bins) for signal slicing. Defaults to 50.
        max_n_sources (int, optional): The maximum number of sources to read from each tar file. Defaults to 4.
        min_power (float, optional): The minimum power of a source to be kept. Defaults to 0 (i.e., no signal is dismissed on account of low power).
        min_count_rate (float, optional): The minimum count rate of a window. Defaults to 0 (i.e., no window is dismissed on account of count rate).
        min_gti_ratio (float, optional): The minimum "good time interval" ratio of a window. Defaults to 0 (i.e., no window is dismissed on account of a low GTI).
        signals_to_extract (list[str], optional): The signals to be extracted. Defaults to ['RATE'].
        output_directory (str | pathlib.Path, optional): The output directory where the files for the full signals will be saved. If `None`, they will be returned directly (instead of just paths to the corresponding files). Defaults to None.
    
    Returns:
        tuple[Float[Array, 'window channel time'], pd.DataFrame, pd.DataFrame, dict, Float[Array, 'window time']]:
        A tuple containing a NumPy array with the signals, a pandas DataFrame of metadata, another one with information
        from the files' headers, and a dictionary with the full signals in every tar.
    
    Raises:
        RuntimeError: If the period of a tar file does not match the previous period(s).
    """

    signals_list = []
    masks_list = []
    last_period = None

    start_list, end_list = [], []
    f_list = []
    names_list = []
    header_df_list = []

    full_signals_dict = {}

    # if the full signals (and masks) are to be saved...
    if output_directory is not None:

        # in case a `str` was passed...
        output_directory = pathlib.Path(output_directory)
            
        full_signals_dir = output_directory / 'full_signals'
        full_signals_masks_dir = full_signals_dir / 'mask'

        full_signals_dir.mkdir(exist_ok=True)
        full_signals_masks_dir.mkdir(exist_ok=True)

    # every tar file...
    for tar_file in files:
        
        # ...is read into 3 `DataFrame`s
        df, header_df, mask_df = read_tar(tar_file, max_n_sources=max_n_sources, signals_to_extract=signals_to_extract)

        # if something was actually read...
        if df is not None:

            df.loc[:, df.columns.get_level_values('data') == 'RATE']

            # checking if the `NaN` values in the `DataFrame` are consistent with the mask; left-hand side is just
            #  `df` with keeping only 'RATE' columns, and removing the `file` level of the column index, so that it can be compared to `mask_df`
            assert (
                df.loc[:, df.columns.get_level_values('data') == 'RATE'].set_axis(df.columns.get_level_values('file').unique(), axis=1).isna()\
                      ==\
                          (~mask_df)
                    ).all().all()
            
            # negative values are clipped to zero
            df = df.clip(lower=0)

            # if there is *any* `NaN` at all (in any column, at any time instant)...
            if df.isna().any().any():

                # ...it's filled in using interpolation.
                df = df.interpolate(method='linear', axis=0)

                if df.isna().any().any():

                    logger.warning(f'"{tar_file}" still has `NaN` values after interpolating, skipping...')
                    
                    continue

            # the period of the raw `DataFrame`
            period = df.index[1] - df.index[0]

            if bin_size == 1:

                binned_df = df
                binned_mask_df = mask_df

            else:
            
                # signals are binned: a *rolling window* is used to bin-collapse together different measurements;
                # the *label* (within the index) of every window is that of the last element in the window; the very
                # first element is dismissed because there is not enough elememts (going backwards) to fill the window
                binned_df = df.rolling(window=bin_size, center=False, step=bin_size).sum().iloc[1:]

                # ...and so are masks
                binned_mask_df = mask_df.rolling(window=bin_size, center=False, step=bin_size).apply(lambda x: np.all(x)).iloc[1:].astype(bool)
                
                # indexes of both should match
                assert (binned_df.index == binned_mask_df.index).all()

            # ---

            # power of the (binned) signals
            power = (binned_df.filter(like='RATE', axis=1)**2).mean(axis=0).droplevel('data')

            # a `Series` with the columns (i.e., signals) whose power is below the threshold
            low_power_columns = power[(power < min_power)].index
            
            # their number
            n_low_power_columns = len(low_power_columns)

            # if every signal has low power...
            if n_low_power_columns == len(power):

                logger.warning(f'every signal has power below {min_power} in "{tar_file}", skipping...')
                
                # ...the file is skipped
                continue

            # at this point not EVERY signal has low power, but those that do have...
            if n_low_power_columns > 0:

                # ...are dropped
                binned_df = binned_df.drop(low_power_columns, axis=1)

                logger.warning(f'dropping {n_low_power_columns} signals in "{tar_file}" due to low power (<{min_power})')

            # ---

            # the period of the *binned* `DataFrame`
            binned_period = binned_df.index[1] - binned_df.index[0]

            # if this is not the first file (`last_period` is `None`)...
            if last_period:

                if period != last_period:

                    logger.critical(f'the period of "{tar_file}" does not match the previous period(s)')
                    
                    raise RuntimeError('Period mismatch')

            # if this IS the first file...
            else:

                window_size_in_seconds = (binned_period * window_size).total_seconds()

                # print(step)
                logger.info(f'Requested a window size of, effectively, duration {window_size_in_seconds} seconds '
                            f'(period is {binned_period.total_seconds()} seconds, step is {(binned_period * step).total_seconds()} seconds)')

            last_period = period

            if len(binned_df) < window_size:

                logger.warning(f'skipping "{tar_file}" because it is too short after (also before?) binning')
                
                continue

            # *full signals* are normalized
            normalized_df = binned_df.apply(lambda x: (x- x.mean()) / x.std())

            # signals are "sliced"
            sliced_signals, sliced_times, n_windows = slice(
                normalized_df, window_size=window_size, step=step, tail_as_window=True, return_n_windows=True)
            
            # `slice` function expects a column multilevel index with 'data' as the second level
            binned_mask_df.columns = pd.MultiIndex.from_tuples([(f, 'OK') for f in binned_mask_df.columns], names=['file', 'data'])
            
            # masks are "sliced" as well
            sliced_masks, _, re_n_windows = slice(
                binned_mask_df, window_size=window_size, step=step, tail_as_window=True, return_n_windows=True)
            assert n_windows == re_n_windows
            assert sliced_signals.shape[::2] == sliced_masks.shape[::2]

            # results for this tar file are stored
            signals_list.append(sliced_signals)
            start_list.append(sliced_times[:, 0])
            end_list.append(sliced_times[:, -1])
            f_list.append(np.full(len(sliced_signals), tar_file.name))
            names_list.append(np.tile(power.index.tolist(), n_windows))
            header_df_list.append(header_df)

            # masks are "squeezed" since they only have a single channel
            masks_list.append(sliced_masks.squeeze())

            # n_signals = len(power.index)
            # i_window = 0
            # i_signal = 0
            # i_signal_col = 0
            # assert np.allclose(sliced_signals[i_window*n_signals + i_signal, i_signal_col, :], normalized_df.iloc[(i_window*step):i_window*step + window_size, 2*i_signal + i_signal_col].values)

            # if a minimum count rate is set...
            if min_count_rate != 0.:

                # count rates
                unnormalized_sliced_signals, *_ = slice(binned_df, window_size=window_size, step=step, tail_as_window=True, return_n_windows=True)

                # this is assuming that the count rate is the first (0-th) channel of the signal
                large_enough_count_rate: Bool[Array, 'window'] = unnormalized_sliced_signals[:, 0, :].mean(axis=1) >= (min_count_rate * bin_size)

                # TODO: this can be avoided if the above check is done right before the above `append`s
                
                # some windows *in the current tar* (`-1`-th element in the lists) are dismissed
                signals_list[-1] = signals_list[-1][large_enough_count_rate]
                start_list[-1] = start_list[-1][large_enough_count_rate]
                end_list[-1] = end_list[-1][large_enough_count_rate]
                f_list[-1] = f_list[-1][large_enough_count_rate]
                names_list[-1] = names_list[-1][large_enough_count_rate]
                masks_list[-1] = masks_list[-1][large_enough_count_rate]

                # for the sake of readability
                n_overall_windows = len(sliced_signals)
                n_large_enough_count_rate_windows = large_enough_count_rate.sum()
                
                # if there are windows to be dismissed here on account of low count rate...
                if n_large_enough_count_rate_windows < n_overall_windows:
                    
                    logger.warning(
                        f'{len(large_enough_count_rate) - large_enough_count_rate.sum()} '
                        f'windows dismissed (out of {n_overall_windows}) on account of low count rate...')

                    # if ALL the windows are to be dismissed...
                    if n_large_enough_count_rate_windows == 0:
                        
                        # ...the header is dismissed as well
                        del header_df_list[-1]

            # if the full signals (and masks) are to be saved...
            if output_directory is not None:

                signals_file = full_signals_dir / f'{tar_file.stem}.parquet'
                masks_file = full_signals_masks_dir / f'{tar_file.stem}.parquet'

                normalized_df.to_parquet(signals_file)
                binned_mask_df.to_parquet(masks_file)

                full_signals_dict[tar_file.name] = signals_file, masks_file

            # if the full signals (and masks) are NOT to be saved...
            else:
                
                # ...just the paths to the corresponding files are stored
                full_signals_dict[tar_file.name] = normalized_df, binned_mask_df
            
    metadata_df = pd.DataFrame(
        {'start': np.hstack(start_list), 'end': np.hstack(end_list), 'tar file': np.hstack(f_list), 'source': np.hstack(names_list)})
    
    # `metadata_df['tar file']`is exactly `np.hstack(f_list)` but it is more convenient to use Pandas' `unique`
    # since it keeps the order of the unique values
    all_headers_df = pd.concat(header_df_list, keys=metadata_df['tar file'].unique().astype(pd.StringDtype))
    
    # (row) indexes are named
    all_headers_df.index.names = ['tar file', 'source']

    # the signals, on one hand, and the masks, on the other, are stacked
    all_signals: Float[Array, 'batch channel length'] = np.vstack(signals_list)
    all_masks: Float[Array, 'batch length'] = np.vstack(masks_list)

    # Good Time Intervals (GTI) are computed
    to_be_kept = (all_masks.mean(axis=-1) >= min_gti_ratio)
    n_to_be_kept = to_be_kept.sum()

    # TODO: compute this just once
    n_overall_windows = len(all_masks)

    n_dismissed_windows = n_overall_windows - n_to_be_kept

    if n_dismissed_windows > 0:

        logger.warning(f'{n_overall_windows - n_to_be_kept} windows dismissed on account of bad time intervals...')

        # tar files will that will not show up in `metadata_df` after keeping only the windows designated by `to_be_kept`
        dismissed_tar_files = set(all_headers_df.index.get_level_values('tar file')) - set(metadata_df.iloc[to_be_kept]['tar file'])
        
        all_headers_df.drop(index=list(dismissed_tar_files), level='tar file', inplace=True)

        for f in dismissed_tar_files:

            del full_signals_dict[f]

            # if saving the full signals (and masks) was requested...
            if output_directory is not None:

                # ...the corresponding files...
                parquet_file = pathlib.Path(f).with_suffix('.parquet')

                # ...are deleted
                (full_signals_dir / parquet_file).unlink()
                (full_signals_masks_dir / parquet_file).unlink()

        # both `all_headers_df` and `metadata_df`, after keeping only the windows designated by `to_be_kept`, "mention" the same tar files
        assert set(all_headers_df.index.get_level_values('tar file')) == set(metadata_df.iloc[to_be_kept]['tar file'])

        return all_signals[to_be_kept], metadata_df.iloc[to_be_kept].reset_index(drop=True), all_headers_df, full_signals_dict, all_masks[to_be_kept]
    
    else:
    
        return all_signals, metadata_df, all_headers_df, full_signals_dict, all_masks


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
    parser.add_argument('-c', '--min-count-rate', type=float, default=None, help='Minimum count rate')
    parser.add_argument('-g', '--min-gti-ratio', type=float, default=None, help='Minimum GTI ratio')
    parser.add_argument('-S', '--signals-to-extract', nargs='+', default=None, help='Signals to be extracted')

    command_line_arguments = parser.parse_args(args)

    # breakpoint()
    
    tar_files = list(command_line_arguments.download_directory.glob('*.tar'))

    # arguments that are `None` and command line-only ones are trimmed
    not_none_args = {
        k: v for k, v in vars(command_line_arguments).items()
        if (v is not None) and (k not in ['download_directory', 'output_directory', 'output_files_stem'])
    }

    # ---

    output_subdirectory = command_line_arguments.output_directory / command_line_arguments.output_files_stem

    # it is created if necessary
    output_subdirectory.mkdir(exist_ok=True)

    # ---

    signals, metadata_df, headers_df, _, masks = tars_to_array(tar_files, **not_none_args, output_directory=output_subdirectory)

    duration = (metadata_df['end'] - metadata_df['start']).iloc[0].total_seconds()

    logger.info(f'{signals.shape[0]} signals of length {signals.shape[-1]} samples ({duration} seconds)')
    
    torch_file = output_subdirectory / 'signals.pt'
    parquet_file = output_subdirectory / 'metadata.parquet'
    yaml_file = output_subdirectory / 'data_preprocessing_parameters.yaml'
    headers_file = output_subdirectory / 'headers.parquet'
    masks_file = output_subdirectory / 'masks.pt'

    torch.save(signals, torch_file) # saved as numpy array
    # torch.save(signals, torch_file, pickle_protocol=pickle.HIGHEST_PROTOCOL) # saved as numpy array
    metadata_df.to_parquet(parquet_file)
    headers_df.to_parquet(headers_file)
    torch.save(masks, masks_file)
    # torch.save(masks, masks_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # for f, (df, mask_df) in full_signals.items():

    #     df.to_parquet(full_signals_dir / f'{pathlib.Path(f).stem}.parquet')
    #     mask_df.to_parquet(full_signals_masks_dir / f'{pathlib.Path(f).stem}.parquet')

    # --- parameters file

    # the default arguments of the function doing the actual work...
    default_args = trapallada.util.func_args_with_defaults(trapallada.xsa.util.tars_to_array)
    
    # ...minus the `files` and `output_directory` arguments
    default_args.pop('files')
    default_args.pop('output_directory')
    
    with open(yaml_file, 'w') as f:
        
        # `default_args` are used only whenever a key is not present in `not_none_args`
        yaml.dump(default_args | not_none_args, f, default_flow_style=False)