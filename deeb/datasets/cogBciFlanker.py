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
import sys
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

FLANKER_BASE_URL = "https://zenodo.org/record/7413650/files/"

download_url="?download=1"

class COGBCIFLANKER(BaseDataset):
    """
    P300 dataset BI2015a from a "Brain Invaders" experiment.

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         BI2015a           28      32        5 NT x 1 T         1s              512Hz           3
        ================ ======= ======= ================ =============== =============== ===========

    **Datasets Description**

    This dataset contains electroencephalographic (EEG) recordings
    of 43 subjects playing to a visual P300 Brain-Computer Interface (BCI)
    videogame named Brain Invaders. The interface uses the oddball paradigm
    on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed
    pseudo-randomly to elicit the P300 response. EEG data were recorded using
    32 active wet electrodes with three conditions: flash duration 50ms, 80ms
    or 110ms. The experiment took place at GIPSA-lab, Grenoble, France, in 2015.
    A full description of the experiment is available at [1]_. The ID of this
    dataset is BI2015a.

    :Investigators: Eng. Louis Korczowski, B. Sc. Martine Cederhout
    :Technical Support: Eng. Anton Andreev, Eng. Grégoire Cattan, Eng. Pedro. L. C. Rodrigues,
                        M. Sc. Violette Gautheret
    :Scientific Supervisor: Ph.D. Marco Congedo

    References
    ----------

    .. [1] Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues, P. L. C.,
           Gautheret, V., & Congedo, M. (2019). Brain Invaders calibration-less P300-based
           BCI with modulation of flash duration Dataset (BI2015a)
           https://hal.archives-ouvertes.fr/hal-02172347
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 30)),
            sessions_per_subject=3, 
            events=dict(Inconsistent=242, Consistent=241),
            code="COG-BCI Flanker",
            interval=[-0.2, 0.8],
            paradigm="n400",
            doi=None,
            dataset_path=None,
            rejection_threshold=None,
            )
    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path, session in zip(file_path_list, [1, 2, 3]):
            session_name = "session_"+str(session)
            if session_name not in sessions.keys():
                sessions[session_name] = {}
            run_name = 'run_1'
            raw_data_path=os.path.join(file_path,"Flanker.set")
            raw = read_raw_eeglab(raw_data_path, preload = True, verbose=False)

            if "Cz" in raw.ch_names:
                raw.drop_channels(['Cz', 'ECG1'])
            else:
                raw.drop_channels(['ECG1'])
            raw.set_montage('standard_1020')

            description_dict = { '20' : 'FLANKER/Start',
                '210' : 'FLANKER Trial/ISI Start',
                '221' : 'FLANKER/Error/ISI',
                '222' : 'FLANKER/Error/FIXI',
                '23' : 'FLANKER/Fixation/Cross',
                '241' : 'FLANKER/Stimulus/cong',
                '2511' : 'FLANKER/Response/Correct/cong',
                '2521' : 'FLANKER/Response/Incorrect/cong',
                '25121' : 'FLANKER/Response/Correct/Feedback/cong',
                '25221' : 'FLANKER/Response/Incorrect/Feedback cong',
                '25321' : 'FLANKER/Missed/Response/Feedback/cong',
                '21' : 'FLANKER/End',
                '242' : 'FLANKER/Stimulus/incong',
                '2512' : 'FLANKER/Response/Correct/incong',
                '2522' : 'FLANKER/Response/Incorrect/incong',
                '25122' : 'FLANKER/Response/Correct/Feedback/incong',
                '25222' : 'FLANKER/Response/Incorrect/Feedback/incong',
                '25322' : 'FLANKER/Missed/Response/Feedback/incong',
                   'boundary': 'boundary'}
            raw.annotations.description=pd.Series(raw.annotations.description).map(description_dict).to_numpy()

            event_ids={
                'FLANKER/Start': 20,
                            'FLANKER Trial/ISI Start': 210,
                            'FLANKER/Error/ISI': 221,
                            'FLANKER/Error/FIXI': 222,
                            'FLANKER/Fixation/Cross': 23,
                            'FLANKER/Stimulus/cong': 241,
                            'FLANKER/Response/Correct/cong': 2511,
                            'FLANKER/Response/Incorrect/cong' : 2521,
                            'FLANKER/Response/Correct/Feedback/cong': 25121,
                            'FLANKER/Response/Incorrect/Feedback cong':25221 ,
                            'FLANKER/Missed/Response/Feedback/cong': 25321,
                            'FLANKER/End': 21,
                            'FLANKER/Stimulus/incong':242 ,
                            'FLANKER/Response/Correct/incong': 2512,
                            'FLANKER/Response/Incorrect/incong': 2522,
                            'FLANKER/Response/Correct/Feedback/incong': 25122,
                            'FLANKER/Response/Incorrect/Feedback/incong': 25222,
                            'FLANKER/Missed/Response/Feedback/incong': 25322
            }
            events, events_ids= mne.events_from_annotations(raw, event_ids, verbose=False)

            events_to_delete = []
            for i in range(len(events) - 1):

                # To delete incorrect congruent events which had incorrect response from the subject
                if events[i][2] == 241 and events[i+1][2] != 2511:
                    events_to_delete.append(i)

                # To delete incorrect incongruent events which had incorrect response from the subject
                elif events[i][2] == 242 and events[i+1][2] != 2512:
                    events_to_delete.append(i)

            events = np.delete(events, events_to_delete, axis=0)
            
            sessions[session_name][run_name] = raw, events
        return sessions

    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        subject_str = f"sub-{subject:02}"
        url = f"{FLANKER_BASE_URL}{subject_str}.zip{download_url}"
        zip_filename = f"{subject_str}.zip{download_url}"

        # download and extract data if needed
        path_zip = dl.data_dl(url, "COGBCIFLANKER2022")
        self.dataset_path=os.path.dirname(os.path.dirname(Path(path_zip.strip(zip_filename))))
        subject_dir = Path(path_zip.strip(zip_filename))/subject_str
        if not subject_dir.exists():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(subject_dir)
        if subject_str in os.listdir(os.path.join(subject_dir, subject_str)):
            subject_dir=Path(os.path.join(subject_dir, subject_str))

        # get paths to relevant files
        session_name="ses"
        session_paths = [
            subject_dir / f"{subject_str}/{session_name}-S{session:1}/eeg" for session in [1, 2, 3]]
        return session_paths
            

            
    

        

    



