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
from . import download as dl
from .base import BaseDataset

class USERDATASET(BaseDataset): 
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 1001)),
            sessions_per_subject=None,
            events={},
            code="User Dataset",
            interval=[-0.2,0.8],
            paradigm="erp",
            doi=None,
            dataset_path=None,
            rejection_threshold=None,
            baseline_correction=True,
            )
    
    def _get_single_subject_data(self, subject):
        """return data for a single subject and session"""
        
        file_path_list = self.data_path(subject)
        sessions = {}
        for session, runs in file_path_list.items():
            sessions[session]={}
            for run in os.listdir(runs):
                sessions[session][run]={}
                run_dir=Path(os.path.join(runs, run))
                for eeg_data in os.listdir(run_dir):
                    raw_data_path=Path(os.path.join(run_dir, eeg_data))
                    raw=mne.io.read_raw_fif(raw_data_path, preload = True, verbose=False)
                    events = mne.find_events(raw, shortest_event=0, verbose=False)
                    sessions[session][run]=raw, events
        return sessions
         
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        "Get path to local copy of a subject data"

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")  
        
        subject="Subject_"+str(subject)
        if subject not in os.listdir(self.dataset_path):
            raise AssertionError(subject, " is not valid")
        
        subject_dir=Path(os.path.join(self.dataset_path, subject))
        ssessions_paths={}

        if len(os.listdir(subject_dir))==0:
            raise AssertionError("Session cannot be Empty")

        else:
            for sess in os.listdir(subject_dir):
                ssessions_paths[sess]=Path(os.path.join(subject_dir, sess))

        return ssessions_paths

        

        


        
        



       
                                



