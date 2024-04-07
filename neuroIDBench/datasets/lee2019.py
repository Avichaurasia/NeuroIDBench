from functools import partialmethod
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat
from . import download as dl
from .base import BaseDataset
import mne
import os

"""
The implementation of this dataset has been sourced from the BDS-3 licensed repository at https://github.com/NeuroTechX/moabb

References
----------
[1] Vinay Jayaram and Alexandre Barachant. MOABB: trustworthy algorithm benchmarking for BCIs. 
Journal of neural engineering 15.6 (2018): 066011. DOI:10.1088/1741-2552
        """

Lee2019_URL = "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100542/"

class Lee2019(BaseDataset):
    """Base dataset class for Lee2019"""
    def __init__(
        self,
        train_run=True,
        test_run=None,
        sessions=(1, 2),
        code_suffix="ERP",
        ):
        for s in sessions:
            if s not in [1, 2]:
                raise ValueError("inexistant session {}".format(s))
        self.sessions = sessions
        super().__init__(
            subjects=list(range(1, 55)),
            sessions_per_subject=2,
            events=dict(Deviant=1),
            code="Lee2019_" + code_suffix,
            interval=[-0.2, 0.8],
            paradigm="erp",
            doi="10.5524/100542",
            dataset_path=None,
            rejection_threshold=None,
            baseline_correction=True,
            )
        self.train_run = train_run
        self.code_suffix=code_suffix
        self.test_run =  self.paradigm == "erp" if test_run is None else test_run

    _scalings = dict(eeg=1e-6, emg=1e-6, stim=1)  # to load the signal in Volts
    
    def _make_raw_array(self, signal, ch_names, ch_type, sfreq, verbose=False):
        """create mne raw array from data

        Parameters:
        ----------
        signal: np.ndarray
            signal data
            ch_names: list
            list of channel names
            ch_type: str
            channel type
            sfreq: float
            sampling frequency
            verbose: bool

        Returns:
        -------
        raw: mne.io.Raw
            raw data
        """
        ch_names = [np.squeeze(c).item() for c in np.ravel(ch_names)]
        if len(ch_names) != signal.shape[1]:
            raise ValueError
        info = create_info(
            ch_names=ch_names, ch_types=[ch_type] * len(ch_names), sfreq=sfreq
        )
        factor = self._scalings.get(ch_type)
        raw = RawArray(data=signal.transpose(1, 0) * factor, info=info, verbose=verbose)
        return raw

    def _get_single_run(self, data):
        """return data for a single run

        Parameters:
        ----------
        data: dict
            dictionary containing the data for a single run

        Returns:
        -------
        raw: mne.io.Raw
            raw data
        """

        sfreq = data["fs"].item()
        file_mapping = {c.item(): int(v.item()) for v, c in data["class"]}
        #self._check_mapping(file_mapping)
    
        # Create RawArray
        raw = self._make_raw_array(data["x"], data["chan"], "eeg", sfreq)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Create stim chan
        event_times_in_samples = data["t"].squeeze()
        event_id = data["y_dec"].squeeze()
        stim_chan = np.zeros(len(raw))
        for i_sample, id_class in zip(event_times_in_samples, event_id):
            stim_chan[i_sample] += id_class
        stim_raw = self._make_raw_array(
            stim_chan[:, None], ["STI 014"], "stim", sfreq, verbose="WARNING"
        )

        # Add EMG and stim channels
        raw = raw.add_channels([stim_raw])
        events=mne.find_events(raw, shortest_event=0, verbose=False)
        return raw, events
    
    def _get_single_subject_data(self, subject):
        """return data for a single subejct

        Parameters:
        ----------
        subject: int
            subject number

        Returns:
        -------
        sessions: dict
            dictionary containing the data for a single subject in the format of {session_name: {run_name: (raw, events)}}  
        """

        sessions = {}
        file_path_list = self.data_path(subject)
        for session in self.sessions:
            if self.train_run or self.test_run:
                mat = loadmat(file_path_list[self.sessions.index(session)])
            session_name = "session_{}".format(session)
            sessions[session_name] = {}
            if self.train_run:
                sessions[session_name]["train"] = self._get_single_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0]
                )
            if self.test_run:
                sessions[session_name]["test"] = self._get_single_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0]
                )
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Get path to local copy of a subject data

        Parameters:
        ----------
        subject: int
            subject number
            path: path to the directory where the data should be downloaded
        
        Returns:
        -------
        subject_paths: list
            list of paths to the local copy of the subject data
        """

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []
        for session in self.sessions:
            url = "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_{3}.mat".format(
                Lee2019_URL, session, subject, self.code_suffix
            )
            data_path = dl.data_dl(url, self.code, path, force_update, verbose)
            self.dataset_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_path))))))
            subject_paths.append(data_path)

        return subject_paths
    
        
