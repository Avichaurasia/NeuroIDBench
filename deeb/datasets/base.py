import abc
import logging
from inspect import signature
from tqdm import tqdm

log = logging.getLogger(__name__)

class BaseDataset(metaclass=abc.ABCMeta):
    """
        The Implementation of this class is based on MOABB (https://github.com/NeuroTechX/moabb) licened under BDS 3-Clause.

        reference:
        ----------
        [1] Vinay Jayaram and Alexandre Barachant. MOABB: trustworthy algorithm benchmarking for BCIs. 
        Journal of neural engineering 15.6 (2018): 066011. DOI:10.1088/1741-2552""
    """
    def __init__(self, subjects, sessions_per_subject, events, code, interval, paradigm, doi=None, dataset_path=None, rejection_threshold=None, 
                 unit_factor=1e6):
        if not isinstance(subjects, list):
            raise ValueError("subjects must be a iterable, like a list")
        #print("subjects", subjects)
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
        #self.subject_list=subject_url_dict

    def get_data(self, subjects=None):
        if subjects is None:
            subjects = self.subject_list
        if not isinstance(subjects, list):
            raise ValueError("subjects must be a list")

        data = {}
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError(f"Invalid subject {subject} given")
            data[subject] = self._get_single_subject_data(subject)
        return data

    def download(self, subject_list=None, path=None, force_update=False, update_path=None, accept=False, verbose=None):
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
        pass

    @abc.abstractmethod
    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        pass
