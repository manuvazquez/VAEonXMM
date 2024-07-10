import pathlib
import tarfile
import re
import tempfile
import shutil
import time

import requests.exceptions

import pandas as pd

# import astropy.io

from astroquery.esa.xmm_newton import XMMNewton

file_types = ['SRCTSR', 'SRSPEC', 'OBSMLI', 'OMSRLI', 'SRCRES']
re_filename = re.compile(r'^P(\d{10})PN([SX]\d{3})(' + '|'.join(file_types) + r')([0-9A-F]+)\.\w+$')

def download_extract_filter(obs: str, output_directory: pathlib.Path | str, keep_tar: bool = False, verbose: bool = True) -> pathlib.Path:
    """
    Downloads and extracts the 'tar' file for a given observation keeping only the files whose name match a specific pattern.

    Args:
        obs (str): The observation identifier.
        output_directory (pathlib.Path | str): The output directory where the files will be saved.
        keep_tar (bool, optional): Whether to keep the downloaded tar file. Defaults to False.
        verbose (bool, optional): Whether to print download messages. Defaults to True.

    Returns:
        pathlib.Path: The path to the output directory where the files are saved.
    """

    # TODO: maybe apply `.zfill(10)` on `obs`

    # for the sake of simplicity
    output_directory = pathlib.Path(output_directory)

    output_subdir = output_directory / obs

    # if it has been downloaded before...
    if output_subdir.exists():

        print(f'found directory "{output_subdir}", skipping...')
        
        # ...return
        return output_subdir
    
    # if it has NOT been downloaded before...
    else:

        # directory is created
        output_subdir.mkdir(exist_ok=True)


    with tempfile.TemporaryDirectory() as tmp_dir:

        # context's descriptor is turned into a `Pathlib`
        tmp_dir_root = pathlib.Path(tmp_dir)
        
        # tar_file = (output_directory if keep_tar else tmp_dir_root) / f'result{obs}.tar'
        tar_file = (output_directory if keep_tar else tmp_dir_root) / f'{obs}.tar'
        

        try:

            # actual download
            XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', verbose=verbose)

        except requests.exceptions.HTTPError:

            print(f'Exception while downloading {obs}, waiting and retrying...')
            time.sleep(1)
            XMMNewton.download_data(obs, filename=tar_file, level='PPS', instname='PN', verbose=verbose)

        
        tmp_dir_extracted = tmp_dir_root / 'extracted'
        
        with tarfile.open(tar_file) as f:

            try:

                files = f.getnames()

            except tarfile.ReadError:

                print(f'There is something wrong with downloaded tar file "{tar_file}"')

                raise
            
            f.extractall(path=tmp_dir_extracted)
        
        # only file (names) matching the pattern are kept...
        m_light_curves = [(f, m) for f in files if (m := re_filename.match(pathlib.Path(f).name))]
        
        # ...and every one of the them is copied into the appropriate directory
        for filename, _ in m_light_curves:
            
            shutil.copy(tmp_dir_extracted / filename, output_subdir)

    return output_subdir


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

