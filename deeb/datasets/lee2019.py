from functools import partialmethod
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset

Lee2019_URL = "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/"

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
            events=dict(Target=1, NonTarget=2),
            code="Lee2019_" + code_suffix,
            interval=[-0.1, 0.9],
            paradigm="p300",
            doi="10.5524/100542",
            dataset_path=None,
            )
        self.train_run = train_run
        self.code_suffix=code_suffix
        self.test_run =  self.paradigm == "p300" if test_run is None else test_run

    _scalings = dict(eeg=1e-6, emg=1e-6, stim=1)  # to load the signal in Volts
    
    def _make_raw_array(self, signal, ch_names, ch_type, sfreq, verbose=False):
        ch_names = [np.squeeze(c).item() for c in np.ravel(ch_names)]
        print("len of channels: ", len(ch_names))
        if len(ch_names) != signal.shape[1]:
            raise ValueError
        info = create_info(
            ch_names=ch_names, ch_types=[ch_type] * len(ch_names), sfreq=sfreq
        )
        factor = self._scalings.get(ch_type)
        raw = RawArray(data=signal.transpose(1, 0) * factor, info=info, verbose=verbose)
        return raw

    def _get_single_run(self, data):
        sfreq = data["fs"].item()
        file_mapping = {c.item(): int(v.item()) for v, c in data["class"]}
        #self._check_mapping(file_mapping)

        

        # Create RawArray
        raw = self._make_raw_array(data["x"], data["chan"], "eeg", sfreq)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Create EMG channels
        #emg_raw = self._make_raw_array(data["EMG"], data["EMG_index"], "emg", sfreq)

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
        return raw
    
    def _get_single_subject_data(self, subject):
        """return data for a single subejct"""

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
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []
        for session in self.sessions:
            url = "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_{3}.mat".format(
                Lee2019_URL, session, subject, self.code_suffix
            )
            data_path = dl.data_dl(url, self.code, path, force_update, verbose)
            subject_paths.append(data_path)

        return subject_paths
    
        
