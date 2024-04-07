import mne
import matplotlib.pyplot as plt
#from sklearn.externals import joblib
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import sys
sys.path.append('.')
import collections
#import spectrum
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import numpy as np
import seaborn as sns
from datetime import datetime as dt
import pandas as pd
import logging
from tqdm import tqdm
from mne.utils import _url_to_local_path, verbose
from .base import Basepipeline
from scipy.signal import welch

log = logging.getLogger(__name__)

class AutoRegressive(Basepipeline):
    """Compute Autoregressive (AR) coefficients from EEG data"""

    def __init__(self, order=6):
        self.order = order

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

        ret = True
        if not ((dataset.paradigm == "erp")) :
            ret = False
        return ret
      
    def _get_features(self, subject_dict, dataset):

        """Compute Autoregressive (AR) coefficients from EEG data.

        This function computes AR coefficients from EEG data. 
        The EEG data for each subject is in the shape of (n_epochs, n_channels, n_times).
        The AR coeffecients are computed using the Yule-Walker method with the maximum 
        likelihood estimation. The order of the AR model isspecified by the user. 

        The resulting DataFrame df contains columns for Subject, session, Event_id, and 
        AR coefficients for each EEG channel, up to a specified order (self.order). 
        
        Parameters:
            - subject_dict (dict): A dictionary containing subject information and EEG data.
            - dataset (object): An object representing the dataset.

        Returns:
        - df (pd.DataFrame): A DataFrame containing AR coefficients for each channel for each epoch.
        """

        df_list = []
        for subject, sessions in tqdm(subject_dict.items(), desc="Computing AR Coeff"):
            for session, runs in sessions.items():
                for run, epochs in runs.items():
                    if not epochs:
                        continue

                    if (dataset.paradigm == "erp"):
                        epochs= epochs['Deviant']
    
                    if (len(epochs)==0):
                            continue
                    epochs_data = epochs.get_data()   
                    for i in range(len(epochs_data)):
                        dictemp = {'subject': subject, "session": session, 'Event_id': list(epochs[i].event_id.values())[0]}
                        for j in range(len(epochs_data[i])):
                            rho, _ = sm.regression.yule_walker(epochs_data[i][j], order=self.order, method="mle")
                            first = epochs.ch_names[j]
                            for d in range(self.order):
                                column_name = f"{first}-AR{d+1}"
                                dictemp[column_name] = rho[d]
                        df_list.append(dictemp)
        df = pd.DataFrame(df_list)
        return df

class PowerSpectralDensity(Basepipeline):
    """Compute Power Spectral Density (PSD) from EEG data"""

    def __init__(self):
        pass

    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm"""

        ret = True
        if not ((dataset.paradigm == "erp")) :
            ret = False
        return ret


    def computing_psd(self, epochs):

        """
        Compute Power Spectral Density (PSD) of EEG epochs using Welch's method.

        Parameters:
            - epochs (mne.Epochs): The EEG epochs for which PSD will be computed.

        Returns:
            - psd_data (array): Array containing the PSD data.
            - freqs (array): Array containing the corresponding frequency values.

        This function calculates the PSD of EEG data using the Welch method. It divides the time
        series data into four time windows with 50% overlap. The PSD is calculated within the frequency
        range from 1 Hz to 50 Hz. The PSD is computed separately for each time window and EEG channel.

        Example usage:
        psd_data, freqs = my_instance.computing_psd(my_epochs)
        """
        tmax=epochs.tmax
        tmin=epochs.tmin
        sfreq=epochs.info['sfreq']
        
        # setting 4 time windows for PSD calculation
        window_duration = (tmax - tmin) / 4
        samples_per_window = int(window_duration * sfreq)

        # Computing PSD with 4 time windows, 50% overlap using welch's method
        spectrum=epochs.compute_psd(method="welch", n_fft=samples_per_window,
            n_overlap=samples_per_window//2, n_per_seg=None, fmin=1, fmax=50, tmin=tmin, tmax=tmax, verbose=False)
        
        return spectrum.get_data(return_freqs=True)
    
    def _get_features(self, subject_dict, dataset):

        """
        Compute Power Spectral Density (PSD) features for EEG data.

        Parameters:
            - subject_dict (dict): A dictionary containing subject information and EEG data.
            - dataset (object): An object representing the dataset.

        Returns:
            - df_psd (pd.DataFrame): A DataFrame containing PSD features.

        This function computes PSD features for EEG data. It iterates through subjects, sessions, and runs
        to calculate PSD for each epoch. The function divides the EEG data into frequency bands (e.g., low,
        alpha, beta, gamma) and computes the average power in each band for each channel. The results are
        stored in a DataFrame.

        Example usage:
        df_psd = my_instance._get_features(my_subject_dict, my_dataset)
        """
        df_psd=pd.DataFrame()
        df_list = []
        FREQ_BANDS = {"low" : [1,10],
                  "alpha" : [10, 13],
                  "beta" : [13,30],
                  "gamma" : [30, 50]}
      
        results = []
        for subject, sessions in tqdm(subject_dict.items(), desc="Computing PSD"):
            for session, runs in sessions.items():
                for run, epochs in runs.items():
                    if not epochs:
                        continue

                    if (dataset.paradigm == "erp"):
                        epochs = epochs['Deviant']
                        
                    if (len(epochs)==0):
                            continue
                    else:
                        result = self.computing_psd(epochs)
                        results.append((result, subject, session, epochs))

        # Computing average band power for each channel
        for result, subject, session, epochs in results:
            psds, freqs = result
            for i in range(len(psds)):
                features = {'subject': subject, 'session': session, 'Event_id': list(epochs[i].event_id.values())[0]}
                for j in range(len(psds[i])):
                    welch_psd=psds[i][j]
                    X=[]
                    for fmin, fmax in FREQ_BANDS.values():
                        
                        # Calculating average power in each frequency band
                        psds_band=welch_psd[(freqs >= fmin) & (freqs < fmax)].mean()
                        X.append(psds_band)
                        
                    channel=epochs.ch_names[j]
                    for d in range(len(X)):
                        band_name=[*FREQ_BANDS][d]
                        colum_name=channel+"-"+band_name
                        features[colum_name]=X[d]
                #data_step = [features]
                df_list.append(features)
        df_psd=pd.DataFrame(df_list)
        return df_psd
    
    
    