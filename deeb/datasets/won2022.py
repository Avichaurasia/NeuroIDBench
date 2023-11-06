import glob
import os
import os.path as osp
from pathlib import Path
import shutil
import zipfile as z
from distutils.dir_util import copy_tree
import mne
import numpy as np
import yaml
from mne.channels import make_standard_montage
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose
import pooch
from pooch import file_hash, retrieve
from requests.exceptions import HTTPError
from scipy.io import loadmat
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset
from collections import OrderedDict
from mne.utils import _url_to_local_path, verbose
import shutil
import io
from pooch import Unzip, retrieve
import mat73

WON2022_BASE_URL="https://ndownloader.figstatic.com/files/"
urls=['32398631', '32398637', '32398625', '32398613', '32398628', '32398631', '32398622', '32398634',
      '32398619', '32398649', '32398685', '32398670', '32398655', '32398679', '32398658', '32398640',
      '32398667', '32398664', '32398652', '32398676', '32398646', '32398643', '32398682', '32398661',
      '32398673', '32398709', '32398688', '32398694', '32398706', '32398697', '32398703', '32398715',
      '32398700', '32398724',  '32398721', '32398712', '32398739', '32398742', '32398718', '32398733',
      '32398757', '32398730', '32398751', '32398766', '32398745', '32398772', '32398748', '32398778',
      '32398754', '32398763', '32398769', '32398787', '32398775', '32398781', '32398760']

class Won2022(BaseDataset):
    """
    P300 dataset BI2015a from a "Brain Invaders" experiment.

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         won2022           43      32        5 NT x 1 T         1s              512Hz           3
        ================ ======= ======= ================ =============== =============== ===========

    """
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 56)),
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=2),
            code="won 2022",
            interval=[-0.1, 0.9],
            paradigm="p300",
            doi=None,
            dataset_path=None,
            rejection_threshold=None,
            baseline_correction=None,
            )
        
    @verbose
    def download_dataset(self, url, sign, subject_str, path=None, force_update=False, verbose=None):
        """
        Download a file from a given URL to a local folder.
        """
        path = Path(dl.get_dataset_path(sign, path))
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
            fname=subject_str,
            path=str(destination.parent),
            progressbar=True,
        )
        return dlpath
    
    def _make_raw_array(self, eeg_data, markers, ch_names, ch_type, sfreq):  
        """create mne raw array from data"""

        chnames = ch_names + ['STI 014']
        ch_types=[ch_type]*len(ch_names)+['stim']
        info = mne.create_info(ch_names=chnames, sfreq=sfreq, ch_types=ch_types, 
                               verbose=False)
        X=np.concatenate((eeg_data, markers[None, :]), axis=0)
        montage=mne.channels.make_standard_montage('biosemi32')
        info.set_montage(montage, match_case=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)
        return raw
    
    def _get_single_run(self, data):
        """return data for a single run"""
        
        eeg_data=np.asarray(data['data'])
        ch_names=[channel['labels'] for channel in data['chanlocs']]
        sfreq=data['srate']
        markers=data['markers_target']
        raw=self._make_raw_array(eeg_data, markers, ch_names, "eeg", sfreq)
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        EEG=mat73.loadmat(file_path_list)
        EEG_train=EEG['test']
        for run in range(0,len(EEG_train)):
            run_name="run_"+str(run+1)
            raw=self._get_single_run(EEG_train[run])
            events=mne.find_events(raw, shortest_event=0, verbose=False)
            sessions[session_name][run_name] = raw, events       
        return sessions
    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        """Get path to local copy of a subject data."""
        
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
       
        # define url and paths
        subject_url_dict={k: v for k, v in zip(self.subject_list, urls)}
        base_url = WON2022_BASE_URL+subject_url_dict[subject]
        subject_str = f"s{subject:02}.mat"
        subject_dir = self.download_dataset(base_url, "Won2022", subject_str)
        self.dataset_path=os.path.dirname(Path(subject_dir.strip(subject_str)))
        return subject_dir
