import pathlib
import tarfile
import re
import tempfile
import shutil

# import pandas as pd

# import astropy.io
# from astropy.table import Table

from astroquery.esa.xmm_newton import XMMNewton

file_types = ['SRCTSR', 'SRSPEC', 'OBSMLI', 'OMSRLI']
re_filename = re.compile(r'^P(\d{10})PN([SX]\d{3})(' + '|'.join(file_types) + r')([0-9A-F]+)\.\w+$')

def download_extract_filter(obs: str, output_directory: pathlib.Path | str, verbose: bool = False) -> pathlib.Path:

    output_subdir = pathlib.Path(output_directory) / obs
    output_subdir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_root = pathlib.Path(tmp_dir)
        tar_file = tmp_dir_root / f'result{obs}.tar'
        XMMNewton.download_data(obs,filename=tar_file, level='PPS', instname='PN', verbose=verbose)
        tmp_dir_extracted = tmp_dir_root / 'extracted'
        with tarfile.open(tar_file) as f:
            files = f.getnames()
            f.extractall(path=tmp_dir_extracted)
        m_light_curves = [(f, m) for f in files if (m := re_filename.match(pathlib.Path(f).name))]
        for filename, _ in m_light_curves:
            shutil.copy(tmp_dir_extracted / filename, output_subdir)

    return output_subdir
