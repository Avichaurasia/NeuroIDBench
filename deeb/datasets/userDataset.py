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
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset

class USERDATASET(BaseDataset): 
    def __init__(self):
        super().__init__(
            subjects=[],
            sessions_per_subject=None,
            events=None,
            code="User Dataset",
            interval=[-0.2,0.8],
            paradigm="p300",
            doi=None,
            dataset_path=None,
            rejection_threshold=None,
            )
    
    def _get_single_subject_data(self, subject):
        """return data for a single subject and session"""
        
        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path, session in zip(file_path_list, [1, 2, 3]):
            session_name = "session_"+str(session)
            if session_name not in sessions.keys():
                sessions[session_name] = {}
            run_name = 'run_1'
            raw_data_path=os.path.join(file_path)
            raw = mne.io.read_raw_fif(raw_data_path, preload = True, verbose=False)
            sessions[session_name][run_name] = raw
        return sessions
         
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        "Get path to local copy of a subject data"

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")  
        subject=str(subject)
        all_subjects_path=os.listdir(self.dataset_path)
        if subject in all_subjects_path:
            subject_dir=Path(os.path.join(self.dataset_path, subject))
        session_name="Session"
        session_paths = [
            subject_dir / f"{subject}/{session_name}_S{session:1}/.fif" for session in os.listdir(os.path.join(self.dataset_path, os.listdir(self.dataset_path)[0]))]
        return session_paths



       
                                



