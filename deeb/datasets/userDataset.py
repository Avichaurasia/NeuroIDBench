#!/usr/bin/env python
import glob
import os
import os.path as osp
import shutil
import zipfile as z
from distutils.dir_util import copy_tree
from pathlib import Path
import mne
import numpy as np
import yaml
from mne.channels import make_standard_montage
from scipy.io import loadmat
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset

BI2015a_URL = "https://zenodo.org/record/3266930/files/"

class User(BaseDataset): 

    def __init__(self):
        super().__init__(
            subjects=None,
            sessions_per_subject=None,
            events=None,
            code="User Dataset",
            interval=[-0.2,0.8],
            paradigm="p300",
            doi=None,
            dataset_path=None,
            rejection_threshold=None,
            )
        print(self.code)
           
    def _get_single_subject_data(self, subject):
        """return data for a single subject and session"""
        
        # file_path_list = self.data_path(subject)
        # all_sessions_data=[]
        # sessions = {}
        return None
        
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        "Get path to local copy of a subject data"

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # define url and paths
        base_url = BI2015a_URL
        subject_str = f"subject_{subject:02}"
        url = f"{base_url}{subject_str}_mat.zip"
        zip_filename = f"{subject_str}.zip"

        # download and extract data if needed
        path_zip = dl.data_dl(url, "BRAININVADERS2015A")
        self.dataset_path=os.path.dirname(os.path.dirname(Path(path_zip.strip(zip_filename))))
        subject_dir = Path(path_zip.strip(zip_filename)) / subject_str
        if not subject_dir.exists():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(subject_dir)
        
        # get paths to relevant files
        session_paths = [
            subject_dir / f"{subject_str}_session_{session:02}.mat" for session in [1, 2, 3]
    ]
        return session_paths
                                



