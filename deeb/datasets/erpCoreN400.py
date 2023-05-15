import glob
import os
import os.path as osp
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
from mne.io import read_raw_eeglab, read_raw
from mne.channels import read_dig_polhemus_isotrak, read_custom_montage
import numpy as np
import pandas as pd
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose
import pooch
from pooch import file_hash, retrieve
from requests.exceptions import HTTPError
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset
from collections import OrderedDict
from mne.utils import _url_to_local_path, verbose
import shutil
import io
from pooch import Unzip, retrieve

ERPCORE_N400_URL = "https://files.osf.io/v1/resources/29xpq/providers/osfstorage/5f24a0c45f705a011461ec9d/?zip="

class ERPCOREN400(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 41)),
            sessions_per_subject=1, 
            events=dict(Inconsistent=222, Consistent=212),
            code="erpcore n400",
            interval=[-0.2, 0.8],
            paradigm="n400",
            doi=None,
            dataset_path=None
            )
    # This function has been sourced from the BDS-3 licensed repository at https://github.com/NeuroTechX/moabb    
    @verbose
    def download_dataset(self, url, sign, path=None, force_update=False, verbose=None):
        """
        This function has been sourced from the BDS-3 licensed repository at https://github.com/NeuroTechX/moabb

        References
        ----------
        [1] Vinay Jayaram and Alexandre Barachant. MOABB: trustworthy algorithm benchmarking for BCIs. 
        Journal of neural engineering 15.6 (2018): 066011. DOI:10.1088/1741-2552
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
            fname="raw_data.zip",
            path=str(destination.parent),
            progressbar=True,
        )

        return dlpath

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        #print(f"file_path_list: {file_path_list}")
        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        run_name = 'run_1'
        raw = read_raw_eeglab(file_path_list, preload = True, verbose=False)
        #raw.annotations.onset = raw.annotations.onset+.026
        raw.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))
        raw.drop_channels(['HEOG_left', 'HEOG_right', 'VEOG_lower'])
        raw.set_montage('standard_1020')
        description_dict = {'111' : 'Prime/Related/L1',
             '112' : 'Prime/Related/L2',
             '121' : 'Prime/Unrelated/L1',
             '122' : 'Prime/Unrelated/L2',
             '211' : 'Target/Related/L1',
             '212' : 'Target/Related/L2',
             '221' : 'Target/Unrelated/L1',
             '222' : 'Target/Unrelated/L2',
             '201' : 'Hit',
             '202' : 'Miss',
             'BAD_seg': 'BAD_seg'      
                   }
        raw.annotations.description=pd.Series(raw.annotations.description).map(description_dict).to_numpy()
        event_ids = {'Prime/Related/L1': 111,
             'Prime/Related/L2': 112,
             'Prime/Unrelated/L1' : 121,
             'Prime/Unrelated/L2' : 122,
             'Target/Related/L1' : 211,
             'Target/Related/L2' : 212,
             'Target/Unrelated/L1' : 221,
             'Target/Unrelated/L2' : 222}
        
        events, event_ids = mne.events_from_annotations(raw, event_ids, verbose=False)

        # Merge events of event_id's "Target/Related because nature of the trails is same, just they
        # came from two diffrerent lists"
        events[events[:, 2] == 211, 2] = 212

       # Merge events of event_id's "Target/Unrelated because nature of the trails is same, just they
        # came from two different lists"
        events[events[:, 2] == 221, 2] = 222
        sessions[session_name][run_name]=raw, events
        return sessions
    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        #subject=str(subject) 
        # else:
        #     if(subject<10):
        #         subject="0"+str(subject)
            #else:
                

        # define url and paths
        url = ERPCORE_N400_URL
        zip_filename = f"raw_data.zip."
        main_directory='raw_data'

         # download and extract data if needed
        path_zip = self.download_dataset(url, "ERPCOREN400")
        self.dataset_path=os.path.dirname(Path(path_zip))
        subject_dir = Path(path_zip.strip(zip_filename))/main_directory
        if not subject_dir.exists():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(subject_dir)

        subject_dir=os.path.join(subject_dir, str(subject))
        raw_data_path = os.listdir(subject_dir)
        for sub in raw_data_path:
            if sub.endswith(".set") and sub.split('.')[0].split('_')[1]=='N400' and len(sub.split('.')[0].split('_'))==2:
                return os.path.join(subject_dir, sub)

        

    



