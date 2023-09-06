from functools import partialmethod
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset
import mne
import os

Lee2019_URL = "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100542/"

class Lee2019(BaseDataset):
    """
    BMI/OpenBMI dataset for P300.

    Dataset summary


        ===========  =======  =======  =================  ===============  ===============  ===========
        Name           #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
        ===========  =======  =======  =================  ===============  ===============  ===========
        Lee2019_ERP       54       62  6900 NT / 1380 T   1s               1000Hz                     2
        ===========  =======  =======  =================  ===============  ===============  ===========

    Dataset from Lee et al 2019 [1]_.

    **Dataset Description**

    EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used
    in the experiment was a BrainAmp (Brain Products; Munich,
    Germany). The channels were nasion-referenced and grounded
    to electrode AFz. Additionally, an EMG electrode recorded from
    each flexor digitorum profundus muscle with the olecranon
    used as reference. The EEG/EMG channel configuration and
    indexing numbers are described in Fig. 1. The impedances of the
    EEG electrodes were maintained below 10 k during the entire
    experiment.

    ERP paradigm
    The interface layout of the speller followed the typical design
    of a row-column speller. The six rows and six columns were
    configured with 36 symbols (A to Z, 1 to 9, and _). Each symbol
    was presented equally spaced. To enhance the
    signal quality, two additional settings were incorporated into
    the original row-column speller design, namely, random-set
    presentation and face stimuli. These additional settings
    help to elicit stronger ERP responses by minimizing adjacency
    distraction errors and by presenting a familiar face image. The
    stimulus-time interval was set to 80 ms, and the inter-stimulus
    interval (ISI) to 135 ms. A single iteration of stimulus presentation
    in all rows and columns was considered a sequence. Therefore,
    one sequence consisted of 12 stimulus flashes. A maximum
    of five sequences (i.e., 60 flashes) was allotted without prolonged
    inter-sequence intervals for each target character. After the end
    of five sequences, 4.5 s were given to the user for identifying, locating,
    and gazing at the next target character. The participant
    was instructed to attend to the target symbol by counting the
    number of times each target character had been flashed.
    In the training session, subjects were asked to copy-spell
    a given sentence, "NEURAL NETWORKS AND DEEP LEARNING"
    (33 characters including spaces) by gazing at the target character
    on the screen. The training session was performed in the offline
    condition, and no feedback was provided to the subject during
    the EEG recording. In the test session, subjects were instructed to
    copy-spell "PATTERN RECOGNITION MACHINE LEARNING"
    (36 characters including spaces), and the real-time EEG data were
    analyzed based on the classifier that was calculated from the
    training session data. The selected character from the subject’s
    current EEG data was displayed in the top left area of the screen
    at the end of the presentation (i.e., after five sequences).
    Per participant, the collected EEG data for the ERP experiment consisted
    of 1,980 and 2,160 trials (samples) for training and test phase, respectively.

    Parameters
    ----------
    train_run: bool (default True)
        if True, return runs corresponding to the training/offline phase (see paper).

    test_run: bool (default: False for MI and SSVEP paradigms, True for ERP)
        if True, return runs corresponding to the test/online phase (see paper). Beware that test_run
        for  MI and SSVEP do not have labels associated with trials: these runs could not be used in
        classification tasks.

    resting_state: bool (default False)
        if True, return runs corresponding to the resting phases before and after recordings (see paper).

    sessions: list of int (default [1,2])
        the list of the sessions to load (2 available).

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """
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
            interval=[-0.2, 0.8],
            paradigm="p300",
            doi="10.5524/100542",
            dataset_path=None,
            rejection_threshold=None,
            )
        self.train_run = train_run
        self.code_suffix=code_suffix
        self.test_run =  self.paradigm == "p300" if test_run is None else test_run

    _scalings = dict(eeg=1e-6, emg=1e-6, stim=1)  # to load the signal in Volts
    
    def _make_raw_array(self, signal, ch_names, ch_type, sfreq, verbose=False):
        """return a RawArray from a signal, channel names and type"""
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
        """return data for a single run"""
        sfreq = data["fs"].item()
        file_mapping = {c.item(): int(v.item()) for v, c in data["class"]}

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
        """Get path to local copy of a subject data"""
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
    
        
