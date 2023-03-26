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

# url_dict=OrderedDict()
# url_dict={1: '32398631', 2: '32398637', 3: '32398625', 4: '32398613', 5: '2398628', 6: '32398631', 7: '32398622', 8: '32398634', 
#         9: '2398619', 10: '32398649', 11: '32398685', 12: '32398670', 13: '32398655', 14: '32398679', 15: '32398658', 16: '32398640',
#         17: '32398667', 18: '32398664', 19: '32398652', 20: '32398676', 21: '32398646', 22: '32398643', 23: '2398682', 24: '2398661',
#         25: '32398673', 26: '32398709', 27: '32398688', 28: '32398694', 29: '32398706', 30: '32398697', 31: '32398703', 32: '32398715',
#         33: '32398700', 34: '32398724', 35: '32398721', 36: '32398712', 37: '32398739', 38: '32398742', 39: '32398718', 40: '32398733',
#         41: '32398757', 42: '32398730', 43: '32398751', 44: '32398766', 45: '32398745', 46: '32398772', 47: '32398748', 48: '32398778',
#         49: '32398754', 50: '32398763', 51: '32398769', 52: '32398787', 53: '32398775', 54: '32398781', 55: '32398760'}
class Won2022(BaseDataset):

    path_to_dataset=" "
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
            )
        #subject_url_dict=OrderedDict
        #updated_url=urls[0:len(self.subject_list)]
          
            
    @verbose
    def download_dataset(self, url, sign, subject_str, path=None, force_update=False, verbose=None):
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
        
        # Channel names are consistent across the train and test EEG data
        #chnames=[channel['labels'] for channel in EEG_train[1]['chanlocs']]
        # chnames=['FP1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 
        #          'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
        #            'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2', 'FZ', 'Cz']
        #chtypes = ['eeg'] * len(chnames) + ['stim']


        # data_train=[np.asarray(EEG_train[n_calib]['data']) for n_calib in range(len(EEG_train))]
        # srate_train=[EEG_train[n_calib]['srate'] for n_calib in range(len(EEG_train))]
        # markers_train=[EEG_train[n_calib]['markers_target'] for n_calib in range(len(EEG_train))]


        # data_test=[np.asarray(EEG_test[n_calib]['data']) for n_calib in range(len(EEG_test))]
        # srate_test=[EEG_test[n_calib]['srate'] for n_calib in range(len(EEG_test))]
        # markers_test=[EEG_test[n_calib]['markers_target'] for n_calib in range(len(EEG_test))]
        #eeg_data=np.

        chnames = ch_names + ['STI 014']
        ch_types=[ch_type]*len(ch_names)+['stim']
        info = mne.create_info(ch_names=chnames, sfreq=sfreq, ch_types=ch_types, 
                               verbose=False)
        #stim=markers
        #print("markers shape", markers.shape)
        X=np.concatenate((eeg_data, markers[None, :]), axis=0)
        #print("X shape", X.shape)

        # make standard montage before read raw data
        montage=mne.channels.make_standard_montage('biosemi32')
        info.set_montage(montage, match_case=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)
        return raw
    
    def _get_single_run(self, data):
        eeg_data=np.asarray(data['data'])
        ch_names=[channel['labels'] for channel in data['chanlocs']]
        sfreq=data['srate']
        markers=data['markers_target']
        #print("markers", markers)
        raw=self._make_raw_array(eeg_data, markers, ch_names, "eeg", sfreq)
        #montage=make_standard_montage('standard_1020')
        #raw.set_montage(montage)
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        print("file path list", file_path_list)
        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        EEG=mat73.loadmat(file_path_list)
        EEG_train=EEG['test']
        #EEG_test=EEG['test']
        for run in range(0,len(EEG_train)):
            run_name="run_"+str(run+1)
            #sessions[session_name][run_name]={} 
            raw=self._get_single_run(EEG_train[run])
            events=mne.find_events(raw, shortest_event=0, verbose=False)

            # print("target events", len(np.where(events[:,2]==1)[0]))
            # print("non-target events", len(np.where(events[:,2]==2)[0]))
            sessions[session_name][run_name] = raw, events       
        return sessions
    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
       
        # define url and paths
        subject_url_dict={k: v for k, v in zip(self.subject_list, urls)}
        base_url = WON2022_BASE_URL+subject_url_dict[subject]
        subject_str = f"s{subject:02}.mat"
        print('subject dir', subject_str)
        #print("Subject_str", subject_str)
        #url = base_url
        #zip_filename = f"{subject_str}.zip"
        #print("zip file dir:", zip_filename)

        # download and extract data if needed
        subject_dir = self.download_dataset(base_url, "Won2022", subject_str)
        #self.dataset_path=os.path.dirname(os.path.dirname(Path(subject_dir.strip(subject_str))))
        self.dataset_path=os.path.dirname(Path(subject_dir.strip(subject_str)))
        print("dataset path", os.path.dirname(Path(subject_dir.strip(subject_str))))
        return subject_dir

# https://ndownloader.figstatic.com/files/3413851
# Subject 1: https://ndownloader.figstatic.com/files/32407757
# Subject 2: https://ndownloader.figstatic.com/files/32398637
# Subject 3: https://ndownloader.figstatic.com/files/32398625
# subject 1: https://doi.org/10.6084/m9.figshare.17707220.v1
# Subject 52; https://doi.org/10.6084/m9.figshare.17701241.v1
