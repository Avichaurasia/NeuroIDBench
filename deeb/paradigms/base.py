import logging
from abc import ABCMeta, abstractmethod

import mne
import numpy as np
import pandas as pd
from collections import OrderedDict
import os
from deeb.paradigms.features import Features
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be \\d+ leaked semaphore objects to clean up at shutdown")

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

    def process_raw(  # noqa: C901
        self, raw, events, dataset, return_epochs=False, return_raws=False
    ):
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
        metadata: pd.DataFrame
            A dataframe containing the metadata

        """

        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        # get events id
        event_id = self.used_events(dataset)
        #print("show event ids", event_id)
        #print("event_id", event_id)
        # find the events, first check stim_channels then annotations
        # stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        # if len(stim_channels) > 0:
        #     print("I am in stim channels")
        #     events = mne.find_events(raw, shortest_event=0, verbose=False)
        # else:
        #     try:
        #         print("I am in annotations")
        #         events, _ = mne.events_from_annotations(
        #             raw, event_id=event_id, verbose=False
        #         )
        #     except ValueError:
        #         log.warning(f"No matching annotations in {raw.filenames}")
        #         return

        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                raw.info["ch_names"], include=self.channels, ordered=True
            )

        # pick events, based on event_id
        try:
            # if type(event_id["Target"]) is list and type(event_id["NonTarget"]) == list:
            #     event_id_new = dict(Target=1, NonTarget=0)
            #     events = mne.merge_events(events, event_id["Target"], 1)
            #     events = mne.merge_events(events, event_id["NonTarget"], 0)
            #     event_id = event_id_new
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        if return_raws:
            raw = raw.pick(picks)
        else:
            # get interval
            tmin = self.tmin + dataset.interval[0]
            if self.tmax is None:
                tmax = dataset.interval[1]
            else:
                tmax = self.tmax + dataset.interval[0]

            X = []
            for bandpass in self.filters:
                fmin, fmax = bandpass
                # filter data
                raw_f = raw.copy().filter(
                    fmin, fmax, method="iir", picks=picks, verbose=False
                )
                # epoch data
                baseline = self.baseline
                # if baseline is not None:
                #     baseline = (
                #         self.baseline[0] + dataset.interval[0],
                #         self.baseline[1] + dataset.interval[0],
                #     )
                #     bmin = baseline[0] if baseline[0] < tmin else tmin
                #     bmax = baseline[1] if baseline[1] > tmax else tmax
                # else:
                #     bmin = tmin
                #     bmax = tmax
                epochs = mne.Epochs(
                    raw_f,
                    events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    proj=False,
                    baseline=None,
                    preload=True,
                    verbose=False,
                    picks=picks,
                    event_repeated="drop",
                    on_missing="ignore",
                )
                # if bmin < tmin or bmax > tmax:
                #     epochs.crop(tmin=tmin, tmax=tmax)
                #if self.resample is not None:
                #   epochs = epochs.resample(self.resample)
                # rescale to work with uV

                #ar = AutoReject(picks=picks, thresh_method='random_search')
                #cleaned_epochs = ar.fit_transform(epochs.copy())
                #cleaned_epochs.apply_baseline(baseline)

                # if return_epochs:
                #     X.append(epochs)
                # else:
                #     X.append(dataset.unit_factor * epochs.get_data())

        #print("Extracted epochs", epochs)
        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in events[:, -1]])
        #print("++++++++++++++Labels+++++++++++++++++++++++++=", len(labels))

        if return_epochs:
            #X = mne.concatenate_epochs(X)
            X=epochs
        elif return_raws:
            X = raw

        # elif len(self.filters) == 1:
        #     # if only one band, return a 3D array
        #     X = X[0]

        else:
            # otherwise return a 4D
            #X = np.array(X).transpose((1, 2, 3, 0))
            X=epochs

        #metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels


    def get_data(self, dataset, subjects=None, return_epochs=False, return_raws=False):
        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        data = dataset.get_data(subjects)
        #del data
        epochs_directory=os.path.join(dataset.dataset_path, "Epochs")
        if not os.path.exists(epochs_directory):
            os.makedirs(epochs_directory)
        else:
            print("Epochs folders already created!")

        self.prepare_process(dataset)
        X = []
        labels = []
        metadata = []
        subject_dict=OrderedDict()
        for subject, sessions in data.items():
            subject_directory=os.path.join(epochs_directory,str(subject))
            if not os.path.exists(subject_directory):
                os.makedirs(subject_directory)
            subject_dict[subject]={}

            for session, runs in sessions.items():
                session_directory=os.path.join(subject_directory, session)
                if not os.path.exists(session_directory):
                    os.makedirs(session_directory)
                subject_dict[subject][session]={}
                
                for run, raw_events in runs.items():
                    raw=raw_events[0]
                    events=raw_events[1]
                    subject_dict[subject][session][run]={}
                    pre_processed_epochs=os.path.join(session_directory, f"{run}_epochs.fif")
                    #met=pd.DataFrame()

                    if return_epochs:
                        if not os.path.exists(pre_processed_epochs):
                            proc = self.process_raw(raw, events, dataset, return_epochs, return_raws)
                            if proc is None:
                            # this mean the run did not contain any selected event
                            # go to next
                                continue
                            x, lbs = proc
                            x.save(pre_processed_epochs, overwrite=True)
                            X.append(x)
                            labels = np.append(labels, lbs, axis=0)
                            #del proc
                        else:
                            x=mne.read_epochs(pre_processed_epochs, preload=True, verbose=False)
                            X.append(x)
                            #labels = np.append(labels, lbs, axis=0)

                    elif return_raws:
                        proc = self.process_raw(raw, events, dataset, return_epochs, return_raws)
                        if proc is None:
                            # this mean the run did not contain any selected event
                            # go to next
                            continue
                        x, lbs = proc
                        X.append(x)
                        labels = np.append(labels, lbs, axis=0)
                        #del proc

                    else:
                        if not os.path.exists(pre_processed_epochs):
                            proc = self.process_raw(raw, events, dataset, return_epochs, return_raws)
                            if proc is None:
                                # this mean the run did not contain any selected event
                                # go to next
                                continue
                            x, lbs = proc
                            x.save(pre_processed_epochs, overwrite=True)
                            X.append(x)
                            labels = np.append(labels, lbs, axis=0)
                        else:
                            x=mne.read_epochs(pre_processed_epochs, preload=True, verbose=False)
                            X.append(x)
                            #del proc
                            #X = np.append(X, x, axis=0) if len(X) else x

                    if(dataset.paradigm=='p300'):
                        erp_paradigm=x['Target']
                    elif(dataset.paradigm=='n400'):
                        erp_paradigm=x['Inconsistent']
                    subject_dict[subject][session][run]=erp_paradigm
                    met = pd.DataFrame(index=range(len(x)))
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    metadata.append(met)
        
        
       
        metadata = pd.concat(metadata, ignore_index=True)
        feat=Features()
        replacement_dict = {v: k for k, v in dataset.event_id.items()}
        if return_epochs:
            X = mne.concatenate_epochs(X, verbose=False)
            # if not os.path.exists(features_directory):
            #     os.makedirs(features_directory)
                # Getting the AR and PSD coeffecients for the dataset 
            features=feat.extract_features(dataset, subject_dict, labels)
            features['Event_id']=features['Event_id'].map(replacement_dict)
                #features.to_csv(os.path.join(features_directory, fname), index=False)
            #else:
             #   features=pd.read_csv(os.path.join(features_directory, fname))
            # print(features['Event_id'].unique())
            # print(dataset.events)
            return X, features, metadata

        elif return_raws:
           return X, metadata
        
        else:
            X = mne.concatenate_epochs(X, verbose=False).get_data()
            # if not os.path.exists(features_directory):
            #     os.makedirs(features_directory)

            # Getting the AR and PSD coeffecients for the dataset 
            features=feat.extract_features(dataset, subject_dict, labels)
            features['Event_id']=features['Event_id'].map(replacement_dict)

                # features.to_csv(os.path.join(features_directory, fname), index=False)
            #print(features['Event_id'].unique())
           # print(dataset.event_id)
            # else:
            #     features=pd.read_csv(os.path.join(features_directory, fname))
            #     print(features['Event_id'].unique())
            #     print(dataset.event_id)
            #gc.collect()
            return X, features, metadata

                    




                    # if not os.path.exists(pre_processed_epochs | pre_processed_epochs_data):
                    #     print("Inside loop")
                    #     proc = self.process_raw(raw, dataset, return_epochs, return_raws)

                    #     if proc is None:
                    #     # this mean the run did not contain any selected event
                    #     # go to next
                    #         continue

                    #     x, lbs = proc
                    #     #print(x.shape)
                    #     if isinstance(x, np.ndarray):
                    #         np.save('my_array.npy', x)
                    #     elif isinstance(x, mne.Epochs):
                    #         x.save(pre_processed_epochs, overwrite=True)
                    #     else:
                    #         #raise ValueError('Invalid input type')
                    #         continue
                    #     #if isinstance(x, mne.Epochs):
                    #      #   x.save(run_data, overwrite=True)

                    # # met["subject"] = subject
                    # # met["session"] = session
                    # # met["run"] = run
                    # # metadata.append(met)
                    
                    # # grow X and labels in a memory efficient way. can be slow
                    #     if return_epochs:
                    #         X.append(x)
                    #     elif return_raws:
                    #         X.append(x)
                    #     else:
                    #         X = np.append(X, x, axis=0) if len(X) else x
                        
                    # else:
                    #     x=mne.read_epochs(run_data, preload=True, verbose=False)
                    #     if return_epochs:
                    #         X.append(x)
                    #     elif return_raws:
                    #         X.append(x)
                    #     else:
                    #         X = np.append(X, x, axis=0) if len(X) else x
                            
        #return X, subject_dict, metadata
