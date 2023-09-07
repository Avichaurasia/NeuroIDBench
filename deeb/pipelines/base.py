import logging
from abc import ABCMeta, abstractmethod

import mne
import numpy as np
import pandas as pd
from collections import OrderedDict
import os
#from deeb.paradigms.features import Features
from tqdm import tqdm
import warnings
from inspect import signature
import gc
log = logging.getLogger(__name__)

class Basepipeline(metaclass=ABCMeta):
     
    @abstractmethod
    def __init__(self):
        pass
        

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

    def get_data(self, dataset, paradigm, subjects=None, return_epochs=False):
        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)
        
        # Get the pre-processed data for the dataset
        X, subjects_dict , metadata = paradigm.get_data(
                dataset, return_epochs)
        data=self._get_features(subjects_dict, dataset)
        return data
 

    @abstractmethod
    def _get_features(self, subject_dict):
        pass
