import pathlib
import time
import random

import requests.exceptions

import pandas as pd

import trapallada.logging as log

# import astropy.io

from astroquery.esa.xmm_newton import XMMNewton

logger = log.setup_logging(__name__)

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
        pathlib.Path: The path to the corresponding *tar* file (regardless of whether it was downloaded or was found already there)
        or `None` if it could not be retrieved.
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
                # XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', verbose=verbose)
                XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', instmode='PrimeFullWindow', verbose=verbose)

                break

            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):

                # print(f'HTTP error while downloading {obs}: {e}')

                attemp += 1

                delay = random.random()*5 + 1

                print(f'Exception while downloading {obs}, waiting {delay} and retrying (attempt #{attemp})...')

                time.sleep(delay)
        
        if attemp == n_attemps:

            logger.error(f'failed to download {obs} after {n_attemps} attempts')

            return None
    
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

