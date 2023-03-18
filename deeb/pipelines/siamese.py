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
import tensorflow as tf
from keras import backend as K

from keras.constraints import max_norm
from keras.layers import (
    Input, Dense, Multiply, Activation, Lambda, Reshape, BatchNormalization,
  LeakyReLU, Flatten, Dropout, Concatenate, Add,
  Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D)
from keras.models import Sequential, Model, load_model, save_model
from keras.callbacks import LearningRateScheduler
#from keras.optimizers import Adam


#   l2, l1, l1_l2,
#    Adam,
#     LearningRateScheduler,
#     plot_model,
#     Sequence,
#     Sequential,
#     Model
# )
import cv2
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from itertools import combinations
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import zscore


class Siamese(Basepipeline):
    def __init__(
        self,
        optimizer="Adam",
        EPOCHS=25,
        batch_size=16,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        #INIT_LR = 0.0005,
        NFFT = 1024,
        FS = 1000,
        CMAP = "viridis",
        FIGSIZE = (0.54, 0.36),
        **kwargs,
    ):
        #NOVERLAP = self.NFFT - 1,
        super().__init__(**kwargs)

        #self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

        self.optimizer = optimizer
        self.EPOCHS = EPOCHS
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path
        self.NFFT = NFFT
        self.FS = FS
        self.CMAP = CMAP
        self.FIGSIZE = FIGSIZE
        self.NOVERLAP = self.NFFT - 1
        #self.INIT_LR = INIT_LR

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

    def get_spectrogram(self, data, fs, nfft, noverlap, figsize, cmap):

    #print(data.shape)
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        #print("I am in function get_spectrogram")
        fig.dpi = 100
        ax.axis('off')
        ax.grid(False)

        pxx, freqs, bins, im = ax.specgram(x=data, Fs=fs, noverlap=noverlap, NFFT=nfft, cmap=cmap)
        print("pxx", pxx.shape)
        print("freqs", len(freqs))
        print("bins", len(bins))
        print("im", im)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar(im)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.major.formatter._useMathText = True
        ax.axis('off')
        ax.grid(False)
        width, height = fig.canvas.get_width_height()
        print("width", width)
        print("height", height)
    # fig is 1000*600 because Figsize is intialized as 10*6 and converted to 1000*600 because fig.dpi =100, 
    #where 600 is height and 1000 is width
        return self.fig2rgb(fig)
    # #print(data.shape)
    #     #print("figsize", figsize)
    #     fig, ax = plt.subplots(1, figsize=figsize, dpi=100)
    #     #print("Figure size 1", fig.get_size_inches())
    #     fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #     #fig.dpi = 100
    #     #print("Figure size 2", fig.get_size_inches())
    #     # width, height = fig.canvas.get_width_height()
    #     # print('Figure width: ', width)
    #     # print('Figure height: ', height)
        
    #     ax.axis('off')
    #     #print("Figure size 3", fig.get_size_inches())
    #     ax.grid(False)
    #     #print("Figure size 4", fig.get_size_inches())
    #     # print('Figure width: ', width)
    #     # print('Figure height: ', height)
        

    #     pxx, freqs, bins, im = ax.specgram(x=data, Fs=fs, noverlap=noverlap, NFFT=nfft, cmap=cmap)
    #     #print("Figure size 5", fig.get_size_inches())
    #     print("pxx", pxx.shape)
    #     print("freqs", len(freqs))
    #     print("bins", len(bins))
    #     print("im", im)

    #     plt.ylabel('Frequency (Hz)')
    #     plt.xlabel('Time (s)')
    #     plt.colorbar(im)
    #     #print("Figure size 6", fig.get_size_inches())

    #     ax.yaxis.set_major_formatter(ScalarFormatter())
    #     #print("Figure size 7", fig.get_size_inches())
    #     ax.yaxis.major.formatter._useMathText = True
    #     #print("Figure size 8", fig.get_size_inches())
    #     ax.axis('off')
    #     #print("Figure size 9", fig.get_size_inches())
    #     ax.grid(False)
    #     #print("Figure size 10", fig.get_size_inches())
    #     return self.fig2rgb(fig)

        #plt.savefig('/content/spectrogram-ar.png', dpi=300, transparent=False, bbox="tight")
        #plt.show()
        
        # fig is 1000*600 because Figsize is intialized as 10*6 and converted to 1000*600 because fig.dpi =100, 
        #where 600 is height and 1000 is width
        #return self.fig2rgb(fig)

    def fig2rgb(self, fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        #print("buf", buf)
        #print("buf shape", len(np.frombuffer(buf, dtype=np.uint8)))
        print(np.frombuffer(buf, dtype=np.uint8).shape)
        width, height = fig.canvas.get_width_height()
        #print("width and height", width , height)
        #plt.show()
        plt.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

    def rgb2gray(self, rgb_img):
        cv_rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(cv_rgb_img, cv2.COLOR_BGR2GRAY)
        return cv_gray

    def min_max_scale(self, spectrogram, f_min, f_max):
        spec_std = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spec_scaled = spec_std * (f_max - f_min) + f_min
        return spec_scaled
    
    
    def build_dataset_old(self, subject_dict, channel_index):
        #subject_dict=OrderedDict()
        count = 0
        X=[]
        Y=[]
        dataset = {}
        for subject, sessions in subject_dict.items():
            for session, runs in sessions.items():
                for run, epochs_path in runs.items():
                    #print("epochs_path", type(epochs_path))
                    #data=mne.read_epochs(epochs_path, verbose=False)
                    epochs=epochs_path['Target']
                    #x = np.array((epoch.get_data(picks='eeg')[:, 0, :]), dtype=np.float)
                    #x=np.array((epochs.get_data(picks='eeg')[:,channel_index,:]), dtype=np.float)
                    x=np.array((epochs.get_data(picks='eeg')), dtype=np.float)
                    x=np.mean(x, axis=1)
                    #print("epochs shape", x.shape)
                    for k in range(x.shape[0]):
                        #print("k", k)
                        print("x[k,:]", x[k,:].shape)

                        # spec = self.get_spectrogram(x[k,:], self.FS, self.NFFT, self.NOVERLAP, self.FIGSIZE, self.CMAP)
                        # #print("grey spec", spec.shape)
                        # gray_spec = self.gb2gray(spec)
                        # scaled_spec = self.min_max_scale(gray_spec, 0.0, 1.0)
                        # if scaled_spec.shape != (36, 54):
                        #     scaled_spec = cv2.resize(scaled_spec, (36, 54))

                        # reshaped_spec = np.reshape(scaled_spec, (scaled_spec.shape[0], scaled_spec.shape[1], 1))
                        reshaped_spec = self.min_max_scale(x[k,:], 0.0, 1.0)
                        print(reshaped_spec.shape)
                        X.append(reshaped_spec)
                        Y.append(count)
            count = count+1
            dataset['x'] = np.array(X)
            dataset['y'] = np.array(Y)
        return dataset
    
    def build_dataset(self, subject_dict, channel_index):
        #subject_dict=OrderedDict()
        count = 0
        X=[]
        Y=[]
        dataset = {}
        for subject, sessions in subject_dict.items():
            for session, runs in sessions.items():
                for run, epochs_path in runs.items():
                    epochs=epochs_path['Target']
                    x=np.array((epochs.get_data(picks='eeg')), dtype=np.float)

                    # Selecting the 100 trails/epochs from each subject
                    x=x[:60]
                    normalized_data=zscore(x, axis=2)
                    X.append(normalized_data)
                    Y.append(np.full(x.shape[0], count))
            count = count+1
        dataset['x'] = np.concatenate(X, axis=0)
        dataset['y'] = np.concatenate(Y, axis=0)
        return dataset
    

    
    def make_pairs(self, data, labels):
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
    
    def euclidean_distance(self, vects):
        (p , q) = vects
        sum = K.square(p - q)
        return K.sqrt(K.maximum(sum, K.epsilon()))
 
    def manhattan_distance(self, vects):
        (p, q) = vects
        return K.abs(p - q)

    #SQRT2 = K.sqrt(K.constant(2.0))

    def hellinger_distance(self, vects):
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
    
    def build_sequential(self, cnn_input):
    #     x = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(cnn_input)
    #     x = BatchNormalization()(x)

    #     x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    #     x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    #     x = BatchNormalization()(x)

    #     x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    #     x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    #     x = BatchNormalization()(x)

    #     x = GlobalAveragePooling2D()(x)

    #     x = Dropout(0.2)(x)

    #     x = Activation("sigmoid")(x)
    #     return x


        x = Conv1D(32, kernel_size=3, activation='relu')(cnn_input)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        return x
    
    #def build_model(self, height, width, channels, model_type, distance_metric):
    def build_model(self, input_shape, model_type, distance_metric):
        model_types = {
        "sequential": self.build_sequential,
        }

        distance_metrics = {
        "euclidean": self.euclidean_distance,
        "manhattan": self.manhattan_distance,
        "hellinger": self.hellinger_distance,
        }

        #input_shape=(height, width, channels)
    
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
        baseLR = 0.001
        power = 1.0
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        return alpha
    
    def model_creation(self, height, weight):

        # Building the Siamese Network with 2 CNN's of spectogram image of 36x54x1 (gray scale)
        #model, cnn_model = self.build_model(36, 54, 1, "sequential", "hellinger")

        # Building the Siamese Network with 2 CNN's network of average time series of all channels
        #model, cnn_model = self.build_model(height, weight, 1, "sequential", "hellinger")
        input_shape=(32,513)
        model, cnn_model = self.build_model(input_shape, "sequential", "euclidean")

        # Compiling the model
        callbacks = [LearningRateScheduler(self.poly_decay)]
        model.compile(loss=self.contrastive_loss, optimizer=self.optimizer)
        return model, cnn_model, callbacks
        
    # Evakluation of the authentication Model
    def evaluate(self, subjects):

        print("I am about to load the dataset")
        # Loading the dataset for the subjects
        dataset=self.build_dataset(subjects, channel_index=8)

        # Gettinbg the data and labels
        data=dataset['x']
        label=dataset['y']

        print(" I am about to split the data")
        # Splitting the data into train and test
        X_train, X_test, y_train, y_test=train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)

        print("I am about to make the pairs")
        # Making the pairs for training and testing
        (pair_train, label_train) = self.make_pairs(X_train, y_train)
        (pair_test, label_test) = self.make_pairs(X_test, y_test)

        # Creating the model
        #model, cnn_model, callbacks = self.model_creation()

        print("I am about to create the model")
        model, cnn_model, callbacks = self.model_creation(1, 513)

        print("I am about to train the model")
        # Training the model
        r = model.fit(
        [pair_train[:, 0], pair_train[:, 1]], label_train[:],
        #validation_data = ([pair_val[:, 0], pair_val[:, 1]], label_val[:]),
        batch_size=self.batch_size, epochs=self.EPOCHS, callbacks=callbacks,
        ) 

        # Saving the model
        #pickle.dump(model, open('./model.pkl', 'wb')) 
        tf.keras.models.save_model(model, './all_channels_time_series.h5', 'wb')

        # Saving the Siamese CNN model
        #pickled_model = pickle.load(open('./model.pkl', 'rb'))
        # custom_objects = {
        #     "contrastive_loss": self.contrastive_loss
        # }
        # pickled_model = tf.keras.models.load_model('./all_channels_time_series.h5', custom_objects=custom_objects)

        # Evaluating the model
        preds = model.predict([pair_test[:, 0], pair_test[:, 1]])
        #model.evaluate([pair_test[:, 0], pair_test[:, 1]], label_test[:], batch_size=self.BATCH_SIZE)
        new_preds= 1.0 - np.squeeze(model.predict([pair_test[:, 0], pair_test[:, 1]]), axis=-1)
        far, frr, eer, acc=self._calculate_far_frr_eer(new_preds, label_test)
        return preds, new_preds
    
    def _calculate_far_frr_eer(self, preds, label_test):
        far = []
        frr = []
        thresholds = np.arange(0,1.01, 0.01)

        #print(thresholds)
        thresholds = np.sort(thresholds)
        #print(thresholds)

        for threshold in thresholds:
            true_positive_count = 0
            true_negative_count = 0
            false_positive_count = 0
            false_negative_count = 0
            for i in range(len(label_test)):
            # Actual Similar
                if label_test[i] == 0:
                    if preds[i] >= threshold:
                        true_positive_count += 1
                    else:
                        false_negative_count += 1
                else:
                    if preds[i] < threshold:
                        true_negative_count += 1
                    else:
                        false_positive_count += 1
            if(threshold==0.01):
                print("FAR at 1%", false_positive_count / (false_positive_count + true_negative_count))
                print("FRR at 1%", false_negative_count / (false_negative_count + true_positive_count))
                print("=============================================")
                print("=============================================")
            
            far.append(false_positive_count / (false_positive_count + true_negative_count))
            frr.append(false_negative_count / (false_negative_count + true_positive_count))

        idx = np.argsort(far)
        #far = np.array(far)[idx]
        far = np.array(far)
        frr = np.array(frr)
        #frr = np.array(frr)[idx]
        tpr = 1 - frr
        auc_tpr=tpr[idx]
        auc_far=far[idx]
        auc = np.trapz(auc_tpr, auc_far)
        hter = (far + frr) / 2.0
        diff = np.absolute(far - frr)
        eer_idx = np.argmin(diff)
        eer=(far[eer_idx] + frr[eer_idx]) / 2.0
        #tpr=1-frr
        print("AUC", auc*100)
        print("EER: ", eer*100)
        print("EER threshold : ", thresholds[eer_idx])
        self._plot_eer(far, frr, thresholds)
        self._plot_roc_auc(far, tpr, auc)
        return far, frr, eer, auc
    
    def _plot_eer(self, far, frr, thresholds):
        fig, ax = plt.subplots()
        ax.plot(thresholds, frr, linestyle = '-', lw=2, alpha=1, label="FRR", color="tab:blue")
        ax.plot(thresholds, far, linestyle = '-', lw=2, alpha=1, label="FAR", color="tab:red")

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Probability')

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.major.formatter._useMathText = True
        ax.legend(frameon=True, loc='best',ncol=1, handlelength=2, framealpha=1, edgecolor="0.8", fancybox=False)
        plt.grid(True, ls="--", lw=0.8)
        plt.tight_layout()
        plt.savefig('./Plots/far_frr.png', dpi=300)
        #plt.savefig('./content/far_frr.png', dpi=300, transparent=False, bbox="tight")
        #plt.show()

    def _plot_roc_auc(self,far, tpr, auc):
        # plt.xticks(np.arange(1,N_WAY+1, 15))
        #tpr = 1 - np.array(frr)
        #auc = np.trapz(tpr, far)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(far, tpr, linestyle = '-', lw=2, alpha=1,label=r'(AUC = %0.2f)' % (auc), 
                color="tab:orange")
        # ax.plot(far, frr, linestyle = '-', lw=2, alpha=1,label="DET", color="tab:red")

        ax.plot([0, 1], [0, 1], "k--", color='r',label="chance level (AUC = 0.5)")
        ax.set_xlabel('False Acceptance Rate (FAR)')
        ax.set_ylabel('True Acceptance Rate (TAR)')

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.major.formatter._useMathText = True
        ax.legend(frameon=True, loc='best',ncol=1, handlelength=2, framealpha=1, edgecolor="0.8", fancybox=False)
        plt.tight_layout()
        plt.grid(True, ls="--", lw=0.8)
        plt.savefig('./Plots/roc.png', dpi=300)
        #plt.show()
    
    def _get_features(self, subjects):
        preds, new_preds=self.evaluate(subjects)
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

        return preds, threshold
    


    





    



        
