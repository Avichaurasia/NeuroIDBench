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
     
    @abstractmethod
    def __init__(self):
        pass
        #pass
        # #print("subjects", subjects)
        # self.subject_list = subjects
        # self.n_sessions = sessions_per_subject
        # self.event_id = events
        # self.code = code
        # self.interval = interval
        # self.paradigm = paradigm
        # self.doi = doi
        # self.unit_factor = unit_factor
        # self.dataset_path=dataset_path
        #self.subject_list=subject_url_dict
        # self.subject_list = subjects_list
        # self.paradigm = paradigm
        # self.dataset = dataset
        # self.dataset_code = dataset_code
        #self.feature_type=feature_type         
        #self.ar_order=ar_order

        

    # @abstractmethod
    # def is_valid(self, dataset):
    #     """Verify the dataset is compatible with the paradigm.

    #     This method is called to verify dataset is compatible with the
    #     paradigm.

    #     This method should raise an error if the dataset is not compatible
    #     with the paradigm. This is for example the case if the
    #     dataset is an ERP dataset for motor imagery paradigm, or if the
    #     dataset does not contain any of the required events.

    #     Parameters
    #     ----------
    #     dataset : dataset instance
    #         The dataset to verify.
    #     """
    #     pass

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

    def get_data(self, dataset, paradigm, subjects=None, return_epochs=False, return_raws=False):
        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        # # Getting the data for the given subjects
        # data = dataset.get_data(subjects)

        # get the data
        X, subjects_dict , metadata = paradigm.get_data(
                dataset, return_epochs)


        #del data
        # epochs_directory=os.path.join(dataset.dataset_path, "Epochs")
        # if not os.path.exists(epochs_directory):
        #     os.makedirs(epochs_directory)
        # else:
        #     print("Epochs folders already created!")

        # self.prepare_process(dataset)

        #data = pd.DataFrame()
        data=self._get_features(subjects_dict, dataset)
        # if (self.feature_type=="ar"):
        #     data = self._get_ar_coeffecients(subject_dict=subjects_dict, ar_order=self.ar_order)

        # elif (self.feature_type=="bandpower"):
        #     data = self._get_average_band_power(subject_dict=subjects_dict)

        # elif (self.feature_type=="siamese"):
        #     data = self._get_siamese_features(subject_dict=subjects_dict)
        return data
 

    @abstractmethod
    def _get_features(self, subject_dict):
        pass

    # @abstractmethod
    # def _get_average_band_power(self, subject_dict):
    #     pass

    # @abstractmethod
    # def _get_siamese_features(self, subject_dict):
    #     pass

    # @abstractmethod
    # def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
    #     pass
