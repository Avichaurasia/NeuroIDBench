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
import os
import pandas as pd
from collections import OrderedDict
import logging
from tqdm import tqdm
from mne.utils import _url_to_local_path, verbose
from deeb.pipelines.base import Basepipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch

log = logging.getLogger(__name__)

class AutoRegressive(Basepipeline):

    def __init__(self, order=6):
        self.order = order

    def is_valid(self, dataset):
        ret = True
        if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")) :
            ret = False
        return ret
      
    def _get_features(self, subject_dict, dataset):
        df_list = []
        print("order", self.order)
        for subject, sessions in tqdm(subject_dict.items(), desc="Computing AR Coeff"):
            for session, runs in sessions.items():
                for run, epochs in runs.items():
                    if not epochs:
                        continue

                    if (dataset.paradigm == "p300"):
                        epochs= epochs['Target']

                    elif (dataset.paradigm == "n400"):
                        epochs = epochs['Inconsistent']
                        
                    if (len(epochs)==0):
                            continue
                    epochs_data = epochs.get_data()   
                    for i in range(len(epochs_data)):
                        dictemp = {'Subject': subject, "session": session, 'Event_id': list(epochs[i].event_id.values())[0]}
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

    def __init__(self):
        pass

    def is_valid(self, dataset):
        ret = True
        if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")) :
            ret = False

        # we should verify list of channels, somehow
        return ret


    def computing_psd(self, epochs):
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

                    if (dataset.paradigm == "p300"):
                        epochs = epochs['Target']
                        
                    elif (dataset.paradigm == "n400"):
                        epochs = epochs['Inconsistent']

                    # Computing PSD for each epoch
                    if (len(epochs)==0):
                            continue
                    else:
                        result = self.computing_psd(epochs)
                        results.append((result, subject, session, epochs))

        # Computing average band power for each channel
        for result, subject, session, epochs in results:
            psds, freqs = result
            for i in range(len(psds)):
                features = {'Subject': subject, 'session': session, 'Event_id': list(epochs[i].event_id.values())[0]}
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
                data_step = [features]
                df_psd=df_psd.append(data_step,ignore_index=True)

        return df_psd
    
    
class StandardScaler_Epoch(BaseEstimator, TransformerMixin):
    """
    Function to standardize the epochs data for the pipeline
    """

    def __init__(self):
        """Init."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_fin = []

        for i in np.arange(X.shape[0]):
            X_p = StandardScaler().fit_transform(X[i])
            X_fin.append(X_p)
        X_fin = np.array(X_fin)

        return X_fin
    
    