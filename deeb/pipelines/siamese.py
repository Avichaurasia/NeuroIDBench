from random import random, seed
import tensorflow as tf
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import (
    Input, Dense, Activation, Lambda, Reshape, BatchNormalization,
  LeakyReLU, Flatten, Dropout, Add,
  MaxPooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D)
from keras.models import Sequential, Model, load_model, save_model
from keras.callbacks import LearningRateScheduler
#from keras.optimizers import Adam
#import tensorflow_addons as tfa
#import tensorflow_addons as tfa
from tensorflow_addons.losses import TripletSemiHardLoss
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from deeb.pipelines.base import Basepipeline


class Siamese():
    def __init__(
        self,
        optimizer="Adam",
        EPOCHS=50,
        batch_size=100,
        verbose=1,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
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

    # def is_valid(self, dataset):
    #     ret = True
    #     if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")):
    #         ret = False

    #     # # check if dataset has required events
    #     # if self.events:
    #     #     if not set(self.events) <= set(dataset.event_id.keys()):
    #     #         ret = False

    #     # we should verify list of channels, somehow
    #     return ret
    
    # Make a function for siamese network embeddings with triplet loss function
    def _siamese_embeddings(self, no_channels, time_steps):

        input = Input((no_channels, time_steps, 1))
        x = Conv2D(128, (1, 15), activation='selu', kernel_initializer='lecun_normal', padding='same')(input)
        x = AveragePooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(32, (1, 15), activation='selu', kernel_initializer='lecun_normal', padding='same')(x)
        x = AveragePooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(16, (1, 15), activation='selu', kernel_initializer='lecun_normal', padding='same')(x)
        x = AveragePooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(8, (1, 15), activation='selu', kernel_initializer='lecun_normal', padding='same')(x)
        x = AveragePooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(4, (1, 15), activation='selu', kernel_initializer='lecun_normal', padding='same')(x)
        x = AveragePooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x = Dense(32, activation=None, kernel_initializer='lecun_normal')(x)
        embeddings =Model(input, x, name="Embedding")
        embeddings.compile(
            optimizer=self.optimizer,
            loss=TripletSemiHardLoss(margin=1.0))
        return embeddings
    
    def _get_features(self, subjects_dict, dataset):

        # Get the no of channels and time steps from the subject dict
        subject_1=subjects_dict[list(subjects_dict.keys())[0]]
        no_channels=subject_1[subject_1.keys().keys()].get_data().shape[1]
        time_steps=subject_1[subject_1.keys().keys()].get_data().shape[2]
        #subject_1=subjects_dict['']

        return self._siamese_embeddings(no_channels, time_steps)
    #     return self._siamese_embeddings(data.shape[1], data.shape[2])
        

