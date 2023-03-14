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
import os
import pandas as pd
from collections import OrderedDict
import logging
from tqdm import tqdm
from mne.utils import _url_to_local_path, verbose
from deeb.pipeline.base import Basepipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
  Input, Dense, Multiply, Activation, Lambda, Reshape, BatchNormalization,
  LeakyReLU, Flatten, Dropout, Concatenate, Add,
  Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D,
)
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import Sequence, plot_model

class Siamese():
    def __init__(self, feature_type):
        self.feature_type = feature_type
        #self._get_siamese_features = self._get_siamese_features()

    def get_spectrogram(data, fs, nfft, noverlap, figsize, cmap):
    #print(data.shape)
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        #print("I am in function get_spectrogram")
        fig.dpi = 100
        ax.axis('off')
        ax.grid(False)

        pxx, freqs, bins, im = ax.specgram(x=data, Fs=fs, noverlap=noverlap, NFFT=nfft, cmap=cmap)
        #print("pxx", pxx.shape)
        #print("freqs", len(freqs))
        #print("bins", len(bins))
        #print("im", im)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar(im)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.major.formatter._useMathText = True
        ax.axis('off')
        ax.grid(False)

        #plt.savefig('/content/spectrogram-ar.png', dpi=300, transparent=False, bbox="tight")
        plt.show()
        
        # fig is 1000*600 because Figsize is intialized as 10*6 and converted to 1000*600 because fig.dpi =100, 
        #where 600 is height and 1000 is width
        return fig2rgb(fig)

    def fig2rgb(fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        width, height = fig.canvas.get_width_height()
        #plt.show()
        plt.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

    def rgb2gray(rgb_img):
        cv_rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(cv_rgb_img, cv2.COLOR_BGR2GRAY)
        return cv_gray

    def min_max_scale(spectrogram, f_min, f_max):
        spec_std = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spec_scaled = spec_std * (f_max - f_min) + f_min
        return spec_scaled
    
    
    def build_dataset(start, end, channel_index):
        #subject_dict=OrderedDict()
        count = 0
        X=[]
        Y=[]
        dataset = {}
        for subject in range(start,end):
            epochs_path=data_path+"/Epochs/"+"Subject_"+str(subject)+".fif"
            #subject_dict[subject]=mne.read_epochs(epochs_path, verbose=False)
            epochs=mne.read_epochs(epochs_path, verbose=False)
            epochs=epochs['Target']
        
            #x = np.array((epoch.get_data(picks='eeg')[:, 0, :]), dtype=np.float)
        
            x=np.array(epochs.get_data()[:,channel_index,:])
            #print("epochs shape", x.shape)

            for k in range(x.shape[0]):
                spec = get_spectrogram(x[k,:], FS, NFFT, NOVERLAP, FIGSIZE, CMAP)
                #print("grey spec", spec.shape)
                gray_spec = rgb2gray(spec)
                scaled_spec = min_max_scale(gray_spec, 0.0, 1.0)
            
            

                if scaled_spec.shape != (36, 54):
                    scaled_spec = cv2.resize(scaled_spec, (36, 54))

                reshaped_spec = np.reshape(scaled_spec, (scaled_spec.shape[0], scaled_spec.shape[1], 1))
                print(reshaped_spec.shape)
                X.append(reshaped_spec)
                Y.append(count)
            count += 1
    #print(X.shape)
        dataset['x'] = np.array(X)
        dataset['y'] = np.array(Y)
        return dataset
    

    
    def make_pairs(data, labels):
        pair_signals = []
        pair_subjects = []

        no_of_classes = len(np.unique(labels))
        idx = [np.where(labels == i)[0] for i in range(0, no_of_classes)]
        #print("oroginal idx", len(idx))
        
        for person in idx:
            # Using the function combinations, possible pair of size 2 for all the epochs inside each subject will be made
            # such (0,1), (0,3)...(890,893)
            positive_combinations = combinations(person, 2)
            #print(positive_combinations)
            #print("Person", len(person))
            #print("combinations", positive_combinations)
            count=0
            for pair in positive_combinations:
                #print(pair)
                #print('Pair', pair)
                #print("===================")
                signal_1_idx = pair[0]
                signal_2_idx = pair[1] 
                current_signal = data[signal_1_idx]

                # Subject_id from the list label will be returned
                subject = labels[signal_1_idx]
                pos_signal = data[signal_2_idx]
                pair_signals.append([current_signal, pos_signal])
                pair_subjects.append(0.0)
        
                neg_idx = np.where(labels != subject)[0]
                #print('negative index', (neg_idx))
                #print("===============================================")
                
                # It will return a epoch data from the person other the person on whom the loop is iterating
                neg_signal = data[np.random.choice(neg_idx)]
                
                pair_signals.append([current_signal, neg_signal])
                pair_subjects.append(1.0)
                count=count+1
            #print("===============================================================")
        
            print('count', count)

        return (np.array(pair_signals), np.array(pair_subjects))
    
    def build_sequential(cnn_input=None):
        x = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(cnn_input)
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

        x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

        x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling2D()(x)

    # x = Dropout(0.2)(x)

        x = Activation("sigmoid")(x)
        return x
    
    def build_model(height, width, channels, model_type, distance_metric):
        model_types = {
        "sequential": build_sequential,
        }

        distance_metrics = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "hellinger": hellinger_distance,
        }

        input_shape=(height, width, channels)
    
    # Siamese Input ----------------------------------------------------------------------------
        siamese_left_input = Input(shape=input_shape)
        siamese_right_input = Input(shape=input_shape)
    # ------------------------------------------------------------------------------------------

    # CNN --------------------------------------------------------------------------------------
        cnn_input = Input(shape=input_shape)
        cnn_output = model_types[model_type](cnn_input)
        cnn_model = Model(inputs=cnn_input, outputs=cnn_output)
    # -------------------------------------------------------------------------------------------

    # Siamese Output-----------------------------------------------------------------------------
        encoded_l = cnn_model(siamese_left_input)
        encoded_r = cnn_model(siamese_right_input)

        distance = Lambda(distance_metrics[distance_metric])([encoded_l, encoded_r])

        drop = Dropout(0.4)(distance)

        dense = Dense(64)(drop)

        siamese_output = Dense(1, activation="sigmoid")(dense)
        siamese_net = Model(inputs=[siamese_left_input, siamese_right_input], outputs=siamese_output)
    # -------------------------------------------------------------------------------------------

        return siamese_net, cnn_model
        
