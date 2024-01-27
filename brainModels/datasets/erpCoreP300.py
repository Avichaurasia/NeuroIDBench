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
from . import download as dl
from .base import BaseDataset
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
from . import download as dl
from .base import BaseDataset
from collections import OrderedDict
from mne.utils import _url_to_local_path, verbose
import shutil
import io
from pooch import Unzip, retrieve

ERPCORE_P300_URL = 'https://files.osf.io/v1/resources/etdkz/providers/osfstorage/5f2492d55f705a010e61b15d/?zip='

class ERPCORENP300(BaseDataset):
    """
    P300 dataset from ERP Core.

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         ERP: N400          40      32        5 NT x 1 T         1s              1024Hz           1
        ================ ======= ======= ================ =============== =============== ===========

        **Datasets Description**

        This dataset included 40 participants, consisting of 25 females and 15 males. 
        The participants were selected from the University of California, Davis community. 
        The mean age of the participants was 21.5 years, with a standard deviation of 2.87. 
        The age range of the participants was between 18 and 30 years. A word pair judgment task 
        was employed to elicit the N400 component in this task. Every experimental trial comprised a red prime 
        word that was subsequently followed by a green target word. Participants were required 
        to indicate whether the target word was semantically related or unrelated to the prime word.

         References
        ----------

        [1] Kappenman, Emily S., et al. "ERP CORE: An open resource for human event-related potential 
        research." NeuroImage 225 (2021): 117465.



    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 41)),
            sessions_per_subject=1, 
            events=dict(Deviant=2, Standard=1),
            code="erpcore p300",
            interval=[-0.2, 0.8],
            paradigm="erp",
            doi=None,
            dataset_path=None,
            rejection_threshold=None,
            baseline_correction=True,
            )
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
        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        run_name = 'run_1'
        raw = read_raw_eeglab(file_path_list, preload = True, verbose=False)
        raw.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))
        raw.drop_channels(['HEOG_left', 'HEOG_right', 'VEOG_lower'])
        raw.set_montage('standard_1020') 
        events, event_ids = mne.events_from_annotations(raw,verbose=False)

        # Getting the targeted events
        P300_events=[event_ids['11'],event_ids['22'],event_ids['33'],event_ids['44'],event_ids['55']]

        fevents=[]
        for i, val in enumerate(events[:,2]):
            if val in [7,6]:
                continue
            if events[i+1,2]==6:
                temp_event=events[i,:]
                if val in P300_events:
                    temp_event[2]=2
                else:
                    temp_event[2]=1
                fevents.append(temp_event)
        fevents = np.array(fevents)
        sessions[session_name][run_name]=raw, fevents
        return sessions
    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number") 
        
        # define url and paths
        url = ERPCORE_P300_URL
        zip_filename = f"raw_data.zip."
        main_directory='raw_data'

         # download and extract data if needed
        path_zip = self.download_dataset(url, "ERPCOREP300")
        self.dataset_path=os.path.dirname(Path(path_zip))
        subject_dir = Path(path_zip.strip(zip_filename))/main_directory
        if not subject_dir.exists():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(subject_dir)

        subject_dir=os.path.join(subject_dir, str(subject))
        raw_data_path = os.listdir(subject_dir)
        for sub in raw_data_path:
            if sub.endswith(".set") and sub.split('.')[0].split('_')[1]=='P3' and len(sub.split('.')[0].split('_'))==2:
                return os.path.join(subject_dir, sub)

        

    



