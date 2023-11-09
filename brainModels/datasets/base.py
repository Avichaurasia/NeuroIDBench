import abc
import logging
from inspect import signature
from tqdm import tqdm
import numpy as np
import os
log = logging.getLogger(__name__)


class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(self, subjects, sessions_per_subject, events, code, interval, paradigm, doi=None, dataset_path=None, rejection_threshold=None, 
                 baseline_correction=True, unit_factor=1e6):
        """"
        Common Parameters for all datasets

        parameters
        ----------
        subjects: List of int
            A list containing subject numbers (or tuples or numpy arrays).

        sessions_per_subject: int
            The number of sessions per subject.

        events: dict of strings
            String codes representing events that match labels in the stimulus channel.
            For ERP codes, you can include:
            - 'Target'
            - 'NonTarget'
            - 'Congruent'
            - 'Incongruent'

        code: string
            A unique identifier for the dataset, which is used in all plots.
            The code should be in CamelCase.

        interval: list with 2 entries
            ERP interval as defined in the dataset description

        paradigm: ['p300','n400']
            Defines the type of dataset. It can be either 'p300' or 'n400'.

        doi: DOI for dataset, optional (for now)
        """
        if not isinstance(subjects, list):
            raise ValueError("subjects must be a iterable, like a list")
        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.code = code
        self.interval = interval
        self.paradigm = paradigm
        self.doi = doi
        self.unit_factor = unit_factor
        self.dataset_path=dataset_path
        self.rejection_threshold=rejection_threshold
        self.baseline_correction=baseline_correction

    def get_data(self, subjects=None):


        """
        Retrieve Data for a List of Subjects

        This function returns the data corresponding to a list of subjects in the following structure:

        data = {
            'subject_id': {
            'session_id': {
            'run_id': run
                }
            }
        }

        The hierarchy starts with subjects, followed by sessions, and then runs. 
        In this context, a session refers to a recording conducted in a 
        single day without removing the EEG cap. A session consists of at least one run, 
        which represents a continuous recording. 
        It is worth noting that some datasets split sessions into multiple runs.

        Parameters
        ----------
        subjects : List of int
            A list containing subject numbers.

        Returns
        -------
        data : Dict
            A dictionary containing the raw data.

        """

        data = {}
        if self.code == "User Dataset":  
            all_subjects=len(np.unique(os.listdir(self.dataset_path)))
            sessions=len(os.listdir(os.path.join(self.dataset_path, os.listdir(self.dataset_path)[0])))
            if subjects is None:
                subjects=np.arange(1, all_subjects+1)
                self.subject_list=np.arange(1, all_subjects+1)
                self.n_sessions=sessions
            else:
                subjects=self.subject_list
        else:
            
            if subjects is None:
                subjects = self.subject_list
            if not isinstance(subjects, list):
                raise ValueError("subjects must be a list")
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError(f"Invalid subject {subject} given")
            data[subject] = self._get_single_subject_data(subject)
        return data

    def download(self, subject_list=None, path=None, force_update=False, update_path=None, accept=False, verbose=None):
        """

        Download All Data from the Dataset

        This function allows you to download all the data from the dataset in a single operation.

        Parameters
        ----------
        subject_list : list of int | None
            A list of subject IDs to download. If set to None, all subjects are downloaded.

        path : None | str
            The location where the data will be stored. If set to None, the function checks for the environment 
            variable or config parameter 'MNE_DATASETS_(dataset)_PATH'. If this doesn't exist, 
            it defaults to the '~/mne_data' directory. If the dataset is not found under the specified path, 
            the data will be automatically downloaded to that folder.

        force_update : bool
            If True, it forces an update of the dataset even if a local copy already exists.

        update_path : bool | None
            If set to True, it configures 'MNE_DATASETS_(dataset)_PATH' in the mne-python config to the provided path. 
            If set to None, the user is prompted for confirmation.

        accept : bool
            If True, it accepts the license terms to proceed with the data download (if any). Default is set to False.

        verbose : bool, str, int, or None
            If not set to None, it overrides the default verbose level (refer to :func:`mne.verbose`).

        """
        if subject_list is None:
            subject_list = self.subject_list
        for subject in subject_list:
            sig = signature(self.data_path)
            params = {
                "subject": subject,
                "path": path,
                "force_update": force_update,
                "update_path": update_path,
                "verbose": verbose,
                "accept": accept,
            } if "accept" in [str(p) for p in sig.parameters] else {
                "subject": subject,
                "path": path,
                "force_update": force_update,
                "update_path": update_path,
                "verbose": verbose,
            }
            self.data_path(**params)

    @abc.abstractmethod
    def _get_single_subject_data(self, subject):

        """
        Return Data for a Single Subject

        This function returns the data for a single subject in the following structure:

        data = {'session_id':
            {'run_id': raw}
            }

        Parameters
        ----------
        subject : int
            The subject number.

        Returns
        -------
        data : Dict
            A dictionary containing the raw data.
        """
        pass

    @abc.abstractmethod
    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):

        """
        Get the Local Path of Subject Data

        This function retrieves the local path to a subject's data.

        Parameters
        ----------
        subject : int
            The subject number.

        path : str | None
            The location to search for the data storage location. If set to None,
            it uses the environment variable or config parameter
            "MNE_DATASETS_(dataset)_PATH." If it doesn't exist, it defaults to the
            "~/mne_data" directory. If the dataset isn't found under the specified
            path, the data is automatically downloaded to that folder.

            force_update : bool
                Force an update of the dataset even if a local copy exists.

            verbose : bool, str, int, or None
                If not None, override the default verbosity level (see :func:`mne.verbose`).

        Returns
        -------
        path : list of str
            The local path to the given data file. This path is contained within a
            list of length one for compatibility.
        """
        pass 
