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
import os
import pandas as pd
from collections import OrderedDict
import logging
from tqdm import tqdm
from mne.utils import _url_to_local_path, verbose
from deeb.pipelines.base import Basepipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# #import tensorflow 
from keras import backend as K

from keras.constraints import max_norm
from keras.layers import (
    Input, Dense, Multiply, Activation, Lambda, Reshape, BatchNormalization,
  LeakyReLU, Flatten, Dropout, Concatenate, Add,
  Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D,
  l2, l1, l1_l2,
   Adam,
    LearningRateScheduler,
    plot_model,
    Sequence,
    Sequential,
    Model
)
import cv2
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from itertools import combinations
from sklearn.model_selection import train_test_split
import pickle


#     Activation,
#     Add,
#     AveragePooling2D,
#     AvgPool2D,
#     Concatenate,
#     Conv2D,
#     Dense,
#     DepthwiseConv2D,
#     Dropout,
#     Flatten,
#     Input,
#     Lambda,
#     LayerNormalization,
#     GlobalAveragePooling2D,
#     MaxPooling2D,
#     Permute,
# )
# from keras.layers.normalization.batch_normalization import BatchNormalization
# from keras.models import Model, Sequential
# from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.layers import (
#   Input, Dense, Multiply, Activation, Lambda, Reshape, BatchNormalization,
#   LeakyReLU, Flatten, Dropout, Concatenate, Add,
#   Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D,
# )
# from tensorflow.keras.regularizers import l2, l1, l1_l2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.utils import Sequence, plot_model
# import tensorflow.keras as keras
# import tensorflow.keras.layers as layers


class Siamese():
    def __init__(
        self,
        loss,
        optimizer="Adam",
        EPOCHS=25,
        batch_size=16,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        NFFT = 160,
        FS = 160,
        CMAP = "viridis",
        FIGSIZE = (10, 6),
        **kwargs,
    ):
        NOVERLAP = self.NFFT - 1,
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.optimizer = optimizer
        self.EPOCHS = EPOCHS
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def get_spectrogram(self, data, fs, nfft, noverlap, figsize, cmap):
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
        #plt.show()
        
        # fig is 1000*600 because Figsize is intialized as 10*6 and converted to 1000*600 because fig.dpi =100, 
        #where 600 is height and 1000 is width
        return self.fig2rgb(fig)

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
    
    
    def build_dataset(self, subject_dict, channel_index):
        #subject_dict=OrderedDict()
        count = 0
        X=[]
        Y=[]
        dataset = {}
        for subject, sessions in subject_dict.items():
            for session, runs in sessions.items():
                for run, epochs_path in runs.items():
                    epochs=mne.read_epochs(epochs_path, verbose=False)
                    epochs=epochs['Target']
                    #x = np.array((epoch.get_data(picks='eeg')[:, 0, :]), dtype=np.float)
                    x=np.array(epochs.get_data()[:,channel_index,:])
                    #print("epochs shape", x.shape)
                    for k in range(x.shape[0]):
                        spec = self.get_spectrogram(x[k,:], self.FS, self.NFFT, self.NOVERLAP, self.FIGSIZE, self.CMAP)
                        #print("grey spec", spec.shape)
                        gray_spec = self.gb2gray(spec)
                        scaled_spec = self.min_max_scale(gray_spec, 0.0, 1.0)
                        if scaled_spec.shape != (36, 54):
                            scaled_spec = cv2.resize(scaled_spec, (36, 54))

                        reshaped_spec = np.reshape(scaled_spec, (scaled_spec.shape[0], scaled_spec.shape[1], 1))
                        print(reshaped_spec.shape)
                        X.append(reshaped_spec)
                        Y.append(count)
            count = count+1
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
    
    def euclidean_distance(vects):
        (p , q) = vects
        sum = K.square(p - q)
        return K.sqrt(K.maximum(sum, K.epsilon()))
 
    def manhattan_distance(vects):
        (p, q) = vects
        return K.abs(p - q)

    #SQRT2 = K.sqrt(K.constant(2.0))

    def hellinger_distance(vects):
        SQRT2 = K.sqrt(K.constant(2.0))
        (p, q) = vects
        # return K.sqrt(K.sum(K.square(p - q), axis=1, keepdims=True)) / SQRT2
        # return K.square(p - q) / SQRT2
        return K.sqrt(K.maximum(K.square(K.sqrt(p) - K.sqrt(q)), K.epsilon())) / SQRT2
    
    def contrastive_loss(self, y_true, y_pred):
        margin = 1.0
        return K.mean((1.0 - y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0.0)))

    def custom_acc(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))
    
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
    
    def build_model(self, height, width, channels, model_type, distance_metric):
        model_types = {
        "sequential": self.build_sequential,
        }

        distance_metrics = {
        "euclidean": self.euclidean_distance,
        "manhattan": self.manhattan_distance,
        "hellinger": self.hellinger_distance,
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
        return siamese_net, cnn_model
    # -------------------------------------------------------------------------------------------  
    
    def poly_decay(self, epoch):
        maxEpochs = self.EPOCHS
        baseLR = Adam.lr
        power = 1.0
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        return alpha
    
    def model_creation(self, subjects):
        model, cnn_model = self.build_model(36, 54, 1, "sequential", "hellinger")
        callbacks = [LearningRateScheduler(self.poly_decay)]
        model.compile(loss=self.contrastive_loss, optimizer=Adam)
        return model, cnn_model, callbacks

    def evaluate(self, subjects):
        model, cnn_model, callbacks = self.model_creation(subjects)
        dataset=self.build_dataset(subjects, channel_index=8)
        data=dataset['x']
        label=dataset['y']
        X_train, X_test, y_train, y_test=train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)
        (pair_train, label_train) = self.make_pairs(X_train, y_train)
        (pair_test, label_test) = self.make_pairs(X_test, y_test)
        r = model.fit(
        [pair_train[:, 0], pair_train[:, 1]], label_train[:],
        #validation_data = ([pair_val[:, 0], pair_val[:, 1]], label_val[:]),
        batch_size=self.batch_size, epochs=self.EPOCHS, callbacks=callbacks,
        ) 
        pickle.dump(model, open('./Siamese/Model/model.pkl', 'wb')) 
        pickled_model = pickle.load(open('./Siamese/Model/model.pkl', 'rb'))
        preds = pickled_model.predict([pair_test[:, 0], pair_test[:, 1]])
        #model.evaluate([pair_test[:, 0], pair_test[:, 1]], label_test[:], batch_size=self.BATCH_SIZE)

        return preds
    
    def predict(self, subjects):
        preds=self.evaluate(subjects)
        avg_sim = []
        avg_nsim = []

        for i, j in enumerate(preds):
            if i%2 == 0:
                avg_sim.append(j[0])
            else:
                avg_nsim.append(j[0])

        threshold = (np.mean(avg_sim) + np.mean(avg_nsim))/2

        print("Similar: ", np.mean(avg_sim))
        print("Dissimilar: ", np.mean(avg_nsim))
        print("Threshold: ", threshold)





    



        
