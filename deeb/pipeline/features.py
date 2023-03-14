import mne
import matplotlib.pyplot as plt
#from sklearn.externals import joblib
from abc import ABCMeta, abstractmethod

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
from deeb.pipeline.base import Basepipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

class AutoRegressive(Basepipeline):

    def __init__(self):
        super().__init__(feature_type='AR')

    def is_valid(self, dataset):
        ret = True
        if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")) :
            ret = False

        # # check if dataset has required events
        # if self.events:
        #     if not set(self.events) <= set(dataset.event_id.keys()):
        #         ret = False

        # we should verify list of channels, somehow
        return ret
    
    def _get_features(self, subject_dict, order=6):
        df_list = []
        #order = 6
        for subject, sessions in tqdm(subject_dict.items(), desc="Computing AR Coeff"):
            for session, runs in sessions.items():
                for run, epochs in runs.items():
                    epochs_data = epochs.get_data()
                    for i in range(len(epochs_data)):
                        dictemp = {'Subject': subject, 'Event_id': list(epochs[i].event_id.values())[0]}
                        for j in range(len(epochs_data[i])):
                            rho, sigma = sm.regression.yule_walker(epochs_data[i][j], order=order, method="mle")
                            first = epochs.ch_names[j]
                            for d in range(order):
                                column_name = f"{first}-AR{d+1}"
                                dictemp[column_name] = rho[d]
                        df_list.append(dictemp)
        df = pd.DataFrame(df_list)
        return df
    
    # @abstractmethod
    # def _get_siamese_features(self, subject_dict):
    #     pass

        #return super()._get_siamese_features(subject_dict)
    
class PowerSpectralDensity(Basepipeline):

    def __init__(self):
        super().__init__(feature_type='bandpower')

    def is_valid(self, dataset):
        ret = True
        if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")) :
            ret = False

        # # check if dataset has required events
        # if self.events:
        #     if not set(self.events) <= set(dataset.event_id.keys()):
        #         ret = False

        # we should verify list of channels, somehow
        return ret


    def computing_psd(self, epochs):
        tmax=epochs.tmax
        tmin=epochs.tmin
        sfreq=epochs.info['sfreq']

        # Using mne.Epochs in-built method compute_psd to calculate PSD using welch's method
        spectrum=epochs.compute_psd(method="welch", n_fft=int(sfreq * (tmax - tmin)),
            n_overlap=0, n_per_seg=None, fmin=1, fmax=50, tmin=tmin, tmax=tmax, verbose=False)
        return spectrum.get_data(return_freqs=True)
    
    def _get_features(self, subject_dict):
        df_psd=pd.DataFrame()
        FREQ_BANDS = {"delta" : [1,4],
                        "theta" : [4,8],
                        "alpha" : [8, 12],
                        "beta" : [12,30],
                        "gamma" : [30, 50]}
        
        results = []
        for subject, sessions in tqdm(subject_dict.items(), desc="Computing PSD"):
            for session, runs in sessions.items():
                for run, epochs in runs.items():
                    result = self.computing_psd(epochs)
                    results.append((result, subject, epochs))

        for result, subject, epochs in results:
            psds, freqs = result
            for i in range(len(psds)):
                features={}
                for j in range(len(psds[i])):
                    welch_psd=psds[i][j]
                    X=[]
                    for fmin, fmax in FREQ_BANDS.values():
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
    
    # def extract_features(self, dataset, subject_dict, labels, ar_order):
    #     df=pd.DataFrame()
    #     df_AR=self.auto_regressive_coeffecients(subject_dict, ar_order)
    #     df_PSD=self.average_band_power(subject_dict)
    #     df=pd.concat([df_AR,df_PSD], axis=1)
    #     #print(df.head())
    #     return df
    
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
    
    