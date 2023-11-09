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
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from brainModels.pipelines.base import Basepipeline

class Siamese():

    """
    A Siamese Neural Network for EEG-based authentication.

    Parameters:
    - EPOCHS (int): The number of training epochs.
    - batch_size (int): The batch size for training.
    - verbose (int): Verbosity mode (0 for silent, 1 for progress bar, 2 for one line per epoch).
    - workers (int): The number of workers to use for data loading.
    
    This class defines a Siamese Neural Network for EEG-based authentication. The network is designed to learn
    representations of EEG data suitable for authentication. It allows customizing the training process, batch size,
    verbosity, random seed, and more.

    Methods:
    - is_valid(dataset): Check if the provided dataset is valid for the given paradigm.

    Example usage:
    siamese = Siamese(EPOCHS=250, batch_size=256, verbose=1, workers=5, random_state=None, validation_split=0.2)
    siamese_model = siamese._siamese_embeddings(no_channels, time_steps)
    siamese_model.fit(training_data, validation_data)
    """
    def __init__(
        self,
        EPOCHS=250,
        batch_size=256,
        verbose=1,
        workers=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.EPOCHS = EPOCHS
        self.batch_size = batch_size
        self.verbose = verbose
        self.workers=workers

    def is_valid(self, dataset):

        """
        Check if a dataset is valid for the given paradigm.

        Parameters:
        - dataset (object): An object representing the EEG dataset.

        Returns:
        - bool: True if the dataset is valid; False otherwise.
        """
        ret = True
        if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")):
            ret = False
        return ret
    
    def _siamese_embeddings(self, no_channels, time_steps):

        """
            
        Siamese Implementation of the Siamese Neural Network for EEG-based authentication in [1]_

        Parameters:
            - no_channels (int): The number of EEG channels.
            - time_steps (int): The number of time steps.

        Returns:
            - SiamModel: A Siamese Neural Network for EEG-based authentication.
            

        The implementation is based on the following paper with some modifications:

        [1] M. Fallahi, T. Strufe and P. Arias-Cabarcos, "BrainNet: Improving Brainwave-based Biometric Recognition with Siamese Networks
        ," 2023 IEEE International Conference on Pervasive Computing and Communications (PerCom), Atlanta, GA, USA, 2023, pp. 53-60, 
        doi: 10.1109/PERCOM56429.2023.10099367.”

        The code has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons

        """

        activef="selu"
        chn=no_channels
        sn=time_steps
        if (sn>512):
            input = tf.keras.layers.Input((chn, sn, 1))
            x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(input)
            x = tf.keras.layers.Conv2D(128, (1, 15), activation=activef, kernel_initializer='lecun_normal')(input)
        else:
            input = tf.keras.layers.Input((chn, sn, 1))
            x = tf.keras.layers.Conv2D(128, (1, 15), activation=activef, kernel_initializer='lecun_normal')(input)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(32, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(16, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(8, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(4, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras. layers.Dense(32, activation=None, kernel_initializer='lecun_normal')(x)
        embedding_network = tf.keras.Model(input, x, name="Embedding")
        embedding_network.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tfa.losses.TripletSemiHardLoss(margin=1.0))
        embedding_network.summary()
        return embedding_network
    
        

