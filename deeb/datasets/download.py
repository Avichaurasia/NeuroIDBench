#!/usr/bin/env python
# coding: utf-8
import json
import os
import os.path as osp
from pathlib import Path
import requests

from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose
import pooch
from pooch import file_hash, retrieve
from requests.exceptions import HTTPError

def get_dataset_path(sign: str, path: str = None) -> str:
    """Returns the dataset path allowing for changes in MNE_DATA config.

    Parameters
    ----------
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.

    Returns
    -------
        path : None | str
        Location of where to look for the data storing location
    """
    print(f'sign: {sign}')
    sign = sign.upper()
    key = f"MNE_DATASETS_{sign}_PATH"
    
    # Set default path if MNE_DATA not already configured
    if os.getenv("MNE_DATA") is None:
        path_def = Path.home()/ "mne_data"
        #path_def=Path('/Volumes/Z Slim')/ "mne_data"
        print(f"MNE_DATA is not already configured. It will be set to "
              f"default location in the home directory - {path_def}\n"
              f"All datasets will be downloaded to this location, if anything is "
              f"already downloaded, please move manually to this location")
        if not path_def.is_dir():
            path_def.mkdir(parents=True)
        os.environ["MNE_DATA"] = str(path_def)
    
    if os.getenv(key) is None:
        os.environ[key] = os.getenv("MNE_DATA")

       
    # Get the final path
    return _get_path(path, key, sign)

@verbose
def data_dl(url, sign, path=None, force_update=False, verbose=None):
    """
    Download file from url to specified path
    This function should replace data_path as the MNE will not support the download
    of dataset anymore. This version is using Pooch.
    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).
    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """
    path = Path(get_dataset_path(sign, path))
    print(f"path: {path}")

    key_dest = f"MNE-{sign.lower()}-data"
    destination = _url_to_local_path(url, path / key_dest)

    destination = str(path) + destination.split(str(path))[1]
    table = {ord(c): "-" for c in ':*?"<>|'}
    destination = Path(str(path) + destination.split(str(path))[1].translate(table))

    if not destination.is_file() or force_update:
        if destination.is_file():
            destination.unlink()
        if not destination.parent.is_dir():
            destination.parent.mkdir(parents=True)
        known_hash = None
    else:
        known_hash = file_hash(str(destination))

    dlpath = retrieve(
        url,
        known_hash,
        fname=Path(url).name,
        path=str(destination.parent),
        progressbar=True,
    )
    return dlpath

