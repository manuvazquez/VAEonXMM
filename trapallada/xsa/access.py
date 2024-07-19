import pathlib
# import tarfile
# import re
# import tempfile
# import shutil
import time
import random

import requests.exceptions

import pandas as pd

# import astropy.io

from astroquery.esa.xmm_newton import XMMNewton

# # SRC: Source
# # TSR: Type of data and processing stage
# # file_types = ['SRCTSR', 'SRSPEC', 'OBSMLI', 'OMSRLI', 'SRCRES']
# file_types = ['SRCTSR']

# # PN: Epic-pn camera
# # re_filename = re.compile(r'^P(\d{10})PN([SX]\d{3})(' + '|'.join(file_types) + r')([0-9A-F]+)\.\w+$')
# re_filename = re.compile(r'^P(\d{10})PN(S\d{3})(' + '|'.join(file_types) + r')([0-9A-F]+)\.\w+$')

# def download_extract_filter(
#         obs: str, output_directory: pathlib.Path | str, keep_tar: bool = False, n_attemps: int = 3, verbose: bool = True) -> pathlib.Path | None:
#     """
#     Downloads and extracts the 'tar' file for a given observation keeping only the files whose name match a specific pattern.
#     Data already processed (through the Pipeline Processing System, or PPS) is returned.

#     Args:
#         obs (str): The observation identifier.
#         output_directory (pathlib.Path | str): The output directory where the files will be saved.
#         keep_tar (bool, optional): Whether to keep the downloaded tar file. Defaults to False.
#         n_attemps (int, optional): Number of attemps to download the file. Defaults to 3.
#         verbose (bool, optional): Whether to print download messages. Defaults to True.

#     Returns:
#         pathlib.Path or None: The path to the output directory where the files were saved if any, or None if nothing was downloaded.
#     """

#     # TODO: maybe apply `.zfill(10)` on `obs`

#     # for the sake of simplicity
#     output_directory = pathlib.Path(output_directory)

#     output_subdir = output_directory / obs

#     # if it has been downloaded before...
#     if output_subdir.exists():

#         print(f'found directory "{output_subdir}", skipping...')
        
#         # ...return
#         # return output_subdir
#         return None
    
#     # if it has NOT been downloaded before...
#     else:

#         # directory is created
#         output_subdir.mkdir(exist_ok=True)

#     # name for the output file (without the path)
#     output_file = f'{obs}.tar'

#     with tempfile.TemporaryDirectory() as tmp_dir:

#         # context's descriptor is turned into a `Pathlib`
#         tmp_dir_root = pathlib.Path(tmp_dir)

#         # the output file in the output directory; it might exist or not...
#         tar_file_in_output_directory = output_directory / output_file

#         # if it does NOT...
#         if not tar_file_in_output_directory.exists():
        
#             # the *tar* file is downloaded either to the output directory or to a temporary one
#             tar_file = (output_directory if keep_tar else tmp_dir_root) / output_file

#             attemp = 0

#             while attemp < n_attemps:
            
#                 try:

#                     # actual download
#                     XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', verbose=verbose)

#                     break

#                 except requests.exceptions.HTTPError as e:

#                     print(f'HTTP error while downloading {obs}: {e}')

#                     delay = random.random() * 5 + 1

#                     print(f'Exception while downloading {obs}, waiting {delay} and retrying (attempt #{attemp})...')
#                     time.sleep(delay)
#                     # XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', verbose=verbose)

#                     attemp += 1
        
#         # it the *tar* file does exist in the output directory...
#         else:

#             print(f'found file "{tar_file_in_output_directory}", extracting...')
            
#             # ...that's the one to be used
#             tar_file = tar_file_in_output_directory

        
#         tmp_dir_extracted = tmp_dir_root / 'extracted'

#         if not tar_file.exists():

#             print(f'Server did not provide the tar file "{tar_file}" (most likely it returned "No results found")')

#             return None
        
#         with tarfile.open(tar_file) as f:

#             try:

#                 files = f.getnames()

#             except tarfile.ReadError:

#                 print(f'There is something wrong with downloaded tar file "{tar_file}"')

#                 raise
            
#             f.extractall(path=tmp_dir_extracted)
        
#         # only file (names) matching the pattern are kept...
#         m_light_curves = [(f, m) for f in files if (m := re_filename.match(pathlib.Path(f).name))]
        
#         # ...and every one of the them is copied into the appropriate directory
#         for filename, _ in m_light_curves:
            
#             shutil.copy(tmp_dir_extracted / filename, output_subdir)

#     return output_subdir

def download_obs(
        obs: str, output_directory: pathlib.Path | str, n_attemps: int = 3, verbose: bool = True) -> pathlib.Path | None:
    """
    Downloads the *tar* file for a given observation. Data already processed (through the Pipeline Processing System, or PPS) is returned.

    Args:
        obs (str): The observation identifier.
        output_directory (pathlib.Path | str): The output directory where the files will be saved.
        n_attemps (int, optional): Number of attemps to download the file. Defaults to 3.
        verbose (bool, optional): Whether to print download messages. Defaults to True.

    Returns:
        pathlib.Path or None: The path to the corresponding *tar* file if downloaded, or None if nothing was downloaded.
    """

    # TODO: maybe apply `.zfill(10)` on `obs`

    # for the sake of simplicity
    output_directory = pathlib.Path(output_directory)

    # name for the output file
    tar_file = output_directory / f'{obs}.tar'

    if not tar_file.exists():
    
        attemp = 0

        while attemp < n_attemps:
        
            try:

                # actual download
                XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', verbose=verbose)

                break

            except requests.exceptions.HTTPError as e:

                # print(f'HTTP error while downloading {obs}: {e}')

                attemp += 1

                print(f'Exception while downloading {obs}, waiting {delay} and retrying (attempt #{attemp})...')
                
                delay = random.random() * 5 + 1

                time.sleep(delay)
    
    # it the *tar* file DOES exist in the output directory...
    else:

        print(f'found file "{tar_file}", skipping...')
        
    return tar_file

def get_observations_in_interval(
        duration_interval: tuple, chunk_size: int = 2_000, columns: tuple[str] = ('observation_id', 'duration')
    ) -> pd.DataFrame:
    """
    Get metadata  from all the observations recorded in a certain interval.

    Args:
        duration_interval (tuple): Minimum and maximum duration

    Returns:
        Metadata for the selected observations
    """

    # for the sake of readability
    duration_min, duration_max = duration_interval

    offset = 0

    # only relevant in the very first iteration
    n_read = chunk_size

    tables_list = []

    while n_read == chunk_size:

        query = f'select TOP {chunk_size} {",".join(columns)} from v_public_observations WHERE duration BETWEEN {duration_min} AND {duration_max} OFFSET {offset}'
        table = XMMNewton.query_xsa_tap(query)
        n_read = len(table)

        tables_list.append(table)
        offset += n_read

    print(f'number of chunks: {len(tables_list)}')

    return pd.concat([e.to_pandas() for e in tables_list], axis=0)

def get_all_tables() -> list[str]:
    """
    Get all the tables offered by XMM-Newton Science Archive.
    """

    return XMMNewton.get_tables()

