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
from inspect import signature
import gc

log = logging.getLogger(__name__)

class Basepipeline(metaclass=ABCMeta):
    def __init__(self, subjects, sessions_per_subject, events, code, interval, paradigm, doi=None, dataset_path=None, unit_factor=1e6):
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
        #self.subject_list=subject_url_dict

    def get_data(self, subject=None):
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

    @abstractmethod
    def _get_ar_coeffecients(self, subject_dict):
        pass

    @abstractmethod
    def _get_average_band_power(self, subject_dict):
        pass

    @abstractmethod
    def _get_siamese_features(self, subject_dict):
        pass

    @abstractmethod
    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        pass
