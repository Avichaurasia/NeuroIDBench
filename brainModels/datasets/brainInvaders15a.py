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
from brainModels.datasets import download as dl
from brainModels.datasets.base import BaseDataset

BI2015a_URL = "https://zenodo.org/record/3266930/files/"

class BrainInvaders2015a(BaseDataset):
    """
    P300 dataset BI2015a from a "Brain Invaders" experiment.

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         BI2015a           43      32        5 NT x 1 T         1s              512Hz           1
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

    Notes
    -----

    .. versionadded:: 0.4.6

    References
    ----------

    .. [1] Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues, P. L. C.,
           Gautheret, V., & Congedo, M. (2019). Brain Invaders calibration-less P300-based
           BCI with modulation of flash duration Dataset (BI2015a)
           https://hal.archives-ouvertes.fr/hal-02172347
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 44)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="Brain Invaders 2015a",
            interval=[-0.2,0.8],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3266929",
            dataset_path=None,
            rejection_threshold=None,
            baseline_correction=None,
            )
        
    def _get_single_subject_data(self, subject):
        """return data for a single subject and session"""
        
        file_path_list = self.data_path(subject)
        all_sessions_data=[]
        sessions = {}
        for file_path, session in zip(file_path_list, [1, 2, 3]):
            session_name = "session_1"
            if session_name not in sessions.keys():
                sessions[session_name] = {}
            run_name = 'run_1'
            chnames = [
                'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9', 'PO10', 'STI 014'
            ]

            chtypes = ["eeg"] * 32 + ["stim"]
            D = loadmat(file_path)["DATA"].T
            S = D[1:33, :] * 1e-6
            stim = D[-2, :] + D[-1, :]
            X = np.concatenate([S, stim[None, :]])
            info = mne.create_info(ch_names=chnames, sfreq=512, ch_types=chtypes, 
                                   verbose=False)

            # make standard montage before read raw data
            raw = mne.io.RawArray(data=X, info=info, verbose=False)
            raw.set_montage(make_standard_montage("standard_1020"))
            all_sessions_data.append(raw)

        # Concetenating the three sessions data since there was no break between each session
        raw_combined=mne.concatenate_raws(all_sessions_data, preload=True, verbose=True)
        events = mne.find_events(raw_combined, shortest_event=0, verbose=False)

        sessions[session_name][run_name]=raw_combined, events        
        return sessions

    
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
                                



