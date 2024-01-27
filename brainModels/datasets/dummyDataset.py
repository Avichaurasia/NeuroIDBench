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
import sys
from mne.channels import read_dig_polhemus_isotrak, read_custom_montage
from mne import Annotations, annotations_from_events, create_info, get_config, set_config
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
from mne.utils import _url_to_local_path, verbose
import shutil
import tempfile
import io

class DummyP300Dataset(BaseDataset):
    """Dummy Dataset for test purpose.

    By default, the dataset has 3 sessions, 10 subjects, and 3 runs per session.

    Parameters
    ----------
    event_list: list or tuple of str
        List of event to generate, default: ("fake1", "fake2", "fake3")
    n_sessions: int, default 2
        Number of session to generate
    n_runs: int, default 3
        Number of runs to generate
    n_subjects: int, default 10
        Number of subject to generate
    paradigm: ['erp']
        Defines what sort of dataset this is
    channels: list or tuple of str
        List of channels to generate, default ("C3", "Cz", "C4")
    duration: float or list of float
        Duration of each run in seconds. If float, same duration for all
        runs. If list, duration for each run.
    n_events: int or list of int
        Number of events per run. If int, same number of events
        for all runs. If list, number of events for each run.
    stim: bool
        If True, pass events through stim channel.
    annotations: bool
        If True, pass events through Annotations."""
    
    def __init__(
    self,
    event_list=("dummy1", "dummy2"),
    n_sessions=3,
    n_runs=3,
    n_subjects=10,
    code="DummyDataset",
    paradigm="erp",
    channels=("C3", "Cz", "C4"),
    sfreq=250,
    duration=120,
    n_events=60,
    stim=True
    ):
        self.n_events = n_events if isinstance(n_events, list) else [n_events] * n_runs
        self.duration = duration if isinstance(duration, list) else [duration] * n_runs
        assert len(self.n_events) == n_runs
        assert len(self.duration) == n_runs
        self.sfreq = sfreq
        event_id = {ev: ii + 1 for ii, ev in enumerate(event_list)}
        self.channels = channels
        self.stim = stim
        #self.annotations = annotations

        super().__init__(
            subjects=list(range(1, n_subjects + 1)),
            sessions_per_subject=n_sessions,
            events=event_id,
            code=code,
            paradigm=paradigm,
            interval=[0, 0.8],
            doi=None,
            dataset_path=None,
        )
        self.n_runs = n_runs
        self.channels = channels
        self.sfreq = sfreq
        self.duration = duration
        self.n_events = n_events
        self.stim = stim

        key = "MNE_DATASETS_{:s}_PATH".format(self.code.upper())
        temp_dir = get_config(key)
        if temp_dir is None or not Path(temp_dir).is_dir():
            temp_dir = tempfile.mkdtemp()
            set_config(key, temp_dir)

    # Function to get single subject data
            
    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = {}
        for session in range(1, self.n_sessions + 1):
            session_name = "session_" + str(session)
            if session_name not in sessions.keys():
                sessions[session_name] = {}
            for run in range(1, self.n_runs + 1):
                run_name = "run_" + str(run)
                if run_name not in sessions[session_name].keys():
                    sessions[session_name][run_name] = {}
                raw = self._generate_raw(
                    self.n_events[run - 1], self.duration[run - 1], stim=self.stim
                )
                events = self._get_events(raw)
                sessions[session_name][run_name] = (raw, events)
        return sessions


    # Function to generate the events
    def _generate_events(self, raw, n_events, stim=True):
        """Generate events for a single run.

        Parameters
        ----------
        raw: mne.io.Raw
            Raw data
        n_events: int
            Number of events to generate
        stim: bool
            If True, pass events through stim channel.
        """
        events = np.zeros((n_events, 3), dtype=int)
        events[:, 0] = np.cumsum(np.random.randint(1, 10, size=n_events))
        events[:, 2] = np.random.randint(1, len(self.events), size=n_events)
        if stim:
            raw.add_events(events, stim_channel="STI 014")
        else:
            raw.add_events(events, stim_channel=None)
        return raw
    
    # Functio to generate the raw data
    def _generate_raw(self, n_events, duration, stim=True):
        """Generate raw data for a single run.

        Parameters
        ----------
        n_events: int
            Number of events to generate
        stim: bool
            If True, pass events through stim channel.
        """
        raw = mne.io.RawArray(
            np.random.randn(len(self.channels), self.sfreq * self.duration),
            mne.create_info(self.channels, self.sfreq, ch_types="eeg"),
        )
        raw.set_montage(make_standard_montage("standard_1005"))
        events = self._generate_events(raw, n_events, stim=stim)

        if self.annotations:
            event_desc = {v: k for k, v in self.event_id.items()}
            if len(events) != 0:
                annotations = annotations_from_events(
                    events, sfreq=self.sfreq, event_desc=event_desc
                )
                annotations.set_durations(self.interval[1] - self.interval[0])
            else:
                annotations = Annotations([], [], [])
            raw.set_annotations(annotations)
        return raw
    
    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        pass

        
        