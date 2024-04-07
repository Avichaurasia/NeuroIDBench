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
import logging
log = logging.getLogger(__name__)

class USERDATASET(BaseDataset): 

    """
    User Dataset

    .. admonition:: Dataset summary

    This is the interface developed to allow users to load their own dataset.
    The dataset should be organized in the following way:
    - A main directory containing subdirectories for each subject
    - Each subject directory should contain subdirectories for each session
    - Each session directory should contain the raw data files for that session
    - The raw data files should be in a standarized format that MNE can read (e.g. .fif) 

    Parameters
    ----------
    dataset_path: str
        Local System Path to the main directory containing the dataset
    rejection_threshold: float
        Threshold for rejection of bad channels
    baseline_correction: bool
        Whether to apply baseline correction to the data
    """

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
        """
        Function to get single subject data

        Parameters:
        ----------
        subject: int
            subject number

        Returns:
        -------
        sessions: dict
            dictionary containing the data for a single subject in the format of {session_name: {run_name: (raw, events)}}

        """
        
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

                    # find the events, first check stim_channels then annotations
                    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
                    if len(stim_channels) > 0:
                        events = mne.find_events(raw, shortest_event=0, verbose=False)
                    else:
                        try:
                            events, _ = mne.events_from_annotations(raw, verbose=False)
                        except ValueError:
                            log.warning(f"No matching annotations in {raw.filenames}")
                            return
                    sessions[session][run]=raw, events
        return sessions
         
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        """
        
        Function to get the path to the data files for a given subject

        Parameters:
        ----------
        subject: int
            subject number
            
        Returns:
        -------
        subject_paths: dict
            dictionary containing the paths to the local copy of the subject data
        """

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

        

        


        
        



       
                                



