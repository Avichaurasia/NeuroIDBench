import logging
from abc import ABCMeta, abstractmethod
import mne
import numpy as np
import pandas as pd
from collections import OrderedDict
import os
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import gc
log = logging.getLogger(__name__)

class BaseParadigm(metaclass=ABCMeta):
    """Base Paradigm."""

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def scoring(self):
        """Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        """
        pass

    @property
    @abstractmethod
    def datasets(self):
        """Property that define the list of compatible datasets"""
        pass

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an ERP dataset for motor imagery paradigm, or if the
        dataset does not contain any of the required events.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """
        pass

    def prepare_process(self, dataset):
        """Prepare processing of raw files

        This function allows to set parameter of the paradigm class prior to
        the preprocessing (process_raw). Does nothing by default and could be
        overloaded if needed.

        Parameters
        ----------

        dataset : dataset instance
        The dataset corresponding to the raw file. mainly use to access
        dataset specific i
        nformation.
        """
        if dataset is not None:
            pass

    def process_raw(self, raw, events, dataset, return_epochs=False):
        """
        Process one raw data file.

        This function apply the preprocessing and eventual epoching on the
        individual run, and return the data, labels and a dataframe with
        metadata.

        metadata is a dataframe with as many row as the length of the data
        and labels.

        Parameters
        ----------
        raw: mne.Raw instance
            the raw EEG data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        return_raws: boolean
            To return raw files and events, to ensure compatibility with braindecode.
            Mutually exclusive with return_epochs

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        
        """
        # get events id
        event_id = self.used_events(dataset)
        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                raw.info["ch_names"], include=self.channels, ordered=True
            )

        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        if dataset.rejection_threshold is not None:
            peak_to_peak_reject=dict(eeg=dataset.rejection_threshold*1e-6)
        else:
            peak_to_peak_reject=None
        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            raw_f = raw.copy().filter(fmin, fmax, picks=picks, verbose=False)
            epochs = mne.Epochs(
                raw_f,
                events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                proj=False,
                baseline=self.baseline,
                reject= peak_to_peak_reject, 
                preload=True,
                verbose=False,
                picks=picks,
                event_repeated="drop",
                on_missing="ignore",
            )
        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in events[:, -1]]) 
        X=epochs
        return X, labels
       
    def get_data(self, dataset, subjects=None, return_epochs=False):

        """
        Return the data for a list of subject.

        return the data, labels and a dataframe with metadata. the dataframe
        will contain at least the following columns

        - subject : the subject indice
        - session : the session indice
        - run : the run indice

        Parameters
        ----------
        dataset:
            A dataset instance.
        subjects: List of int
            List of subject number
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        
        Returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        subject_dict: Python dictionary containing the preprocessed 
                      data for each subject    
        metadata: pd.DataFrame
            A dataframe containing the metadata.
        """
        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)
        replacement_dict = {v: k for k, v in dataset.event_id.items()}

        # This returns the raw mne data for the given number of subjects in the form of dictionary
        data = dataset.get_data(dataset.subject_list)
        self.prepare_process(dataset)
        X = []
        labels = []
        metadata = []
        subject_dict=OrderedDict()
        for subject, sessions in tqdm(data.items(), desc="Extracting epochs"):
            subject_dict[subject]={}
            for session, runs in sessions.items():
                subject_dict[subject][session]={}
                for run, raw_events in runs.items():
                    raw=raw_events[0]
                    events=raw_events[1]
                    subject_dict[subject][session][run]={}     
                    proc = self.process_raw(raw, events, dataset, return_epochs)
                    x, lbs = proc
                    if (proc is None) or (len(x)==0):
                    # this mean the run did not contain any selected event
                    # go to next
                        continue
                    subject_dict[subject][session][run]=x
                    X.append(x)
                    labels = np.append(labels, lbs, axis=0)
                    met = pd.DataFrame(index=range(len(x)))
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    met["event_id"] = x.events[:, 2].astype(int).tolist()
                    met["event_id"]=met["event_id"].map(replacement_dict)
                    metadata.append(met)
        metadata = pd.concat(metadata, ignore_index=True)
        if return_epochs:
            X = mne.concatenate_epochs(X, verbose=False)
            return X, subject_dict, metadata  
        else:
            X = mne.concatenate_epochs(X, verbose=False).get_data()
            return X, subject_dict, metadata

                    



