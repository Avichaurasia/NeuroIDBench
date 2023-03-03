import mne
import matplotlib.pyplot as plt
#from sklearn.externals import joblib
import sys
sys.path.append('.')
import collections
#import spectrum
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import numpy as np
from mne.decoding import Vectorizer
import seaborn as sns
from datetime import datetime as dt
import os
import pandas as pd
from collections import OrderedDict
from itertools import islice
import logging
from tqdm import tqdm
#import multiprocessing
from mne.utils import _url_to_local_path, verbose
from deeb.paradigms.psd_utils import computing_psd

#from deeb.paradigms.base import BaseParadigm
log = logging.getLogger(__name__)

class Features():

    
    def auto_regressive_coeffecients(self, subject_dict):
        df_list = []
        order = 6
        for subject, sessions in tqdm(subject_dict.items(), desc="Computing AR Coeffecients"):
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

    def computing_psd(self, epochs):
        tmax=epochs.tmax
        tmin=epochs.tmin
        sfreq=epochs.info['sfreq']

        # Using mne.Epochs in-built method compute_psd to calculate PSD using welch's method
        spectrum=epochs.compute_psd(method="welch", n_fft=int(sfreq * (tmax - tmin)),
            n_overlap=0, n_per_seg=None, fmin=1, fmax=50, tmin=tmin, tmax=tmax, verbose=False)

        return spectrum.get_data(return_freqs=True)
    
    def average_band_power(self, subject_dict):
        df_psd=pd.DataFrame()
        FREQ_BANDS = {"delta" : [1,4],
                        "theta" : [4,8],
                        "alpha" : [8, 12],
                        "beta" : [12,30],
                        "gamma" : [30, 50]}
        
        #pool = multiprocessing.Pool()
        results = []

        for subject, sessions in subject_dict.items():
            for session, runs in sessions.items():
                for run, epochs in runs.items():
                    result = self.computing_psd(epochs)
                    results.append((result, subject, epochs))

        for result, subject, epochs in tqdm(results, desc="Computing PSD"):
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

    # def average_band_power(self, subject_dict):
    #     df_psd=pd.DataFrame()
    #     FREQ_BANDS = {"delta" : [1,4],
    #                     "theta" : [4,8],
    #                     "alpha" : [8, 12],
    #                     "beta" : [12,30],
    #                     "gamma" : [30, 50]}
        
    #     for subject, sessions in tqdm(subject_dict.items(), desc="Computing PSD"):
    #         for session, runs in sessions.items():
    #             for run, epochs in runs.items():
    #                 tmax=epochs.tmax
    #                 tmin=epochs.tmin
    #                 sfreq=epochs.info['sfreq']

    #                 # Using mne.Epochs in-built method compute_psd to calculate PSD using welch's method
    #                 spectrum=epochs.compute_psd(method="welch", n_fft=int(sfreq * (tmax - tmin)),
    #                     n_overlap=0, n_per_seg=None, fmin=1, fmax=50, tmin=tmin, tmax=tmax, verbose=False)

    #                 psds, freqs=spectrum.get_data(return_freqs=True)
    #                 for i in range(len(psds)):
    #                     features={}
    #                     #event_id = list(epochs.event_id.values())[0]
    #                     #features = {'Subject': subject, 'Event_id_PSD': event_id}
    #                     for j in range(len(psds[i])):
    #                         welch_psd=psds[i][j]
    #                         X=[]
    #                         for fmin, fmax in FREQ_BANDS.values():
    #                             psds_band=welch_psd[(freqs >= fmin) & (freqs < fmax)].mean()
    #                             X.append(psds_band)
            
    #                         #features['Subject']=subject
    #                         #features['Event_id_PSD']=list(epochs[i].event_id.values())[0]
        
    #                         channel=epochs.ch_names[j]
    #                         for d in range(len(X)):
    #                             band_name=[*FREQ_BANDS][d]
    #                             colum_name=channel+"-"+band_name
    #                             features[colum_name]=X[d]
    #                     data_step = [features]
    #                     df_psd=df_psd.append(data_step,ignore_index=True)

    #     return df_psd
    
    def extract_features(self, dataset, subject_dict, labels):
        df=pd.DataFrame()
        df_AR=self.auto_regressive_coeffecients(subject_dict)
        df_PSD=self.average_band_power(subject_dict)
        df=pd.concat([df_AR,df_PSD], axis=1)
        #print(df.head())
        return df
    
    #import multiprocessing

    # def average_band_power(self, subject_dict):
    #     df_psd=pd.DataFrame()
    #     FREQ_BANDS = {"delta" : [1,4],
    #                     "theta" : [4,8],
    #                     "alpha" : [8, 12],
    #                     "beta" : [12,30],
    #                     "gamma" : [30, 50]}
        
    #     pool = multiprocessing.Pool()
    #     results = []

    #     for subject, sessions in subject_dict.items():
    #         for session, runs in sessions.items():
    #             for run, epochs in runs.items():
    #                 result = pool.apply_async(computing_psd, (epochs,))
    #                 results.append((result, subject, epochs))

    #     pool.close()
    #     pool.join()

    #     for result, subject, epochs in tqdm(results, desc="Computing PSD"):
    #         psds, freqs = result.get()
    #         for i in range(len(psds)):
    #             features={}
    #             for j in range(len(psds[i])):
    #                 welch_psd=psds[i][j]
    #                 X=[]
    #                 for fmin, fmax in FREQ_BANDS.values():
    #                     psds_band=welch_psd[(freqs >= fmin) & (freqs < fmax)].mean()
    #                     X.append(psds_band)
            
    #                 channel=epochs.ch_names[j]
    #                 for d in range(len(X)):
    #                     band_name=[*FREQ_BANDS][d]
    #                     colum_name=channel+"-"+band_name
    #                     features[colum_name]=X[d]
    #             data_step = [features]
    #             df_psd=df_psd.append(data_step,ignore_index=True)

    #     return df_psd

    
    
  
    
    



