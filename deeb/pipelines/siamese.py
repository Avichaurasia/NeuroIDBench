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
import tensorflow_addons as tfa
#from tensorflow_addons.losses import TripletSemiHardLoss
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from deeb.pipelines.base import Basepipeline


class Siamese():
    # def __init__(self) -> None:
    #     pass
    def __init__(
        self,
        #optimizer="Adam",
        EPOCHS=250,
        batch_size=256,
        verbose=1,
        workers=5,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        #NOVERLAP = self.NFFT - 1,
        super().__init__(**kwargs)

        #self.loss = loss
        # if optimizer == "Adam":
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

        #self.optimizer = optimizer
        self.EPOCHS = EPOCHS
        self.batch_size = batch_size
        self.verbose = verbose
        self.workers=workers
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def is_valid(self, dataset):
        ret = True
        if not ((dataset.paradigm == "p300") | (dataset.paradigm == "n400")):
            ret = False

        # # check if dataset has required events
        # if self.events:
        #     if not set(self.events) <= set(dataset.event_id.keys()):
        #         ret = False

        # we should verify list of channels, somehow
        return ret
    
    # # This function has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons
    def _siamese_embeddings(self, no_channels, time_steps):
        activef="selu"
        chn=no_channels
        sn=time_steps

        print("chn", chn)
        print("sn", sn)

        
        #x = tf.keras.layers.BatchNormalization()(input)
        if (sn>512):
            input = tf.keras.layers.Input((chn, sn, 1))
            x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(input)
            x = tf.keras.layers.Conv2D(128, (1, 15), activation=activef, kernel_initializer='lecun_normal')(input)
        else:
            input = tf.keras.layers.Input((chn, sn, 1))
            x = tf.keras.layers.Conv2D(128, (1, 15), activation=activef, kernel_initializer='lecun_normal')(input)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        #x = keras.layers.MaxPooling2D(pool_size=(1, 4))(x)
        x = tf.keras.layers.Conv2D(32, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        #x = keras.layers.AveragePooling2D(pool_size=(1, 5))(x)
        x = tf.keras.layers.Conv2D(16, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # = keras.layers.AveragePooling2D(pool_size=(1,5))(x)

        x = tf.keras.layers.Conv2D(8, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Conv2D(4, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)


        x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras. layers.Dense(32, activation=None, kernel_initializer='lecun_normal')(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        embedding_network = tf.keras.Model(input, x, name="Embedding")
        embedding_network.compile(
        optimizer=tf.keras.optimizers.Adam(),
        #0.001, clipnorm=1.
        loss=tfa.losses.TripletSemiHardLoss(margin=1.0))
    #siamese.summary()
        embedding_network.summary()
        return embedding_network
    
        

