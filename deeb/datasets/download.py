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

# This function has been sourced from the BDS-3 licensed repository at https://github.com/NeuroTechX/moabb
def get_dataset_path(sign: str, path: str = None) -> str:
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

# This function has been sourced from the BDS-3 licensed repository at https://github.com/NeuroTechX/moabb
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

# # This function is from https://github.com/cognoma/figshare (BSD-3-Clause)
# def fs_issue_request(method, url, headers, data=None, binary=False):
#     """Wrapper for HTTP request
#     Parameters
#     ----------
#     method : str
#         HTTP method. One of GET, PUT, POST or DELETE
#     url : str
#         URL for the request
#     headers: dict
#         HTTP header information
#     data: dict
#         Figshare article data
#     binary: bool
#         Whether data is binary or not
#     Returns
#     -------
#     response_data: dict
#         JSON response for the request returned as python dict
#     """
#     if data is not None and not binary:
#         data = json.dumps(data)

#     response = requests.request(method, url, headers=headers, data=data)

#     try:
#         response.raise_for_status()
#         try:
#             response_data = json.loads(response.text)
#         except ValueError:
#             response_data = response.content
#     except HTTPError as error:
#         print("Caught an HTTPError: {}".format(error))
#         print("Body:\n", response.text)
#         raise

#     return response_data


# def fs_get_file_list(article_id, version=None):
#     """List all the files associated with a given article.
#     Parameters
#     ----------
#     article_id : str or int
#         Figshare article ID
#     version : str or id, default is None
#         Figshare article version. If None, selects the most recent version.
#     Returns
#     -------
#     response : dict
#         HTTP request response as a python dict
#     """
#     fsurl = "https://api.figshare.com/v2"
#     if version is None:
#         url = fsurl + "/articles/{}/files".format(article_id)
#         headers = {"Content-Type": "application/json"}
#         response = fs_issue_request("GET", url, headers=headers)
#         return response
#     else:
#         url = fsurl + "/articles/{}/versions/{}".format(article_id, version)
#         headers = {"Content-Type": "application/json"}
#         request = fs_issue_request("GET", url, headers=headers)
#         return request["files"]


# def fs_get_file_hash(filelist):
#     """Returns a dict associating figshare file id to MD5 hash
#     Parameters
#     ----------
#     filelist : list of dict
#         HTTP request response from fs_get_file_list
#     Returns
#     -------
#     response : dict
#         keys are file_id and values are md5 hash
#     """
#     return {str(f["id"]): "md5:" + f["supplied_md5"] for f in filelist}


# def fs_get_file_id(filelist):
#     """Returns a dict associating filename to figshare file id
#     Parameters
#     ----------
#     filelist : list of dict
#         HTTP request response from fs_get_file_list
#     Returns
#     -------
#     response : dict
#         keys are filname and values are file_id
#     """
#     return {f["name"]: str(f["id"]) for f in filelist}


# def fs_get_file_name(filelist):
#     """Returns a dict associating figshare file id to filename
#     Parameters
#     ----------
#     filelist : list of dict
#         HTTP request response from fs_get_file_list
#     Returns
#     -------
#     response : dict
#         keys are file_id and values are file name
#     """
#     return {str(f["id"]): f["name"] for f in filelist}

# # if __name__ == "__main__":
# #     #print(Path.home() / "mne_data")
# #     Draschkow2018_URL = "https://zenodo.org/record/3266930/files/"
# #     #BI2015a_URL = "https://zenodo.org/record/3266930/files/"
# #     url = f"{Draschkow2018_URL}subject_02_mat.zip"
# #     path_zip = data_dl(url, "Draschkow2018")