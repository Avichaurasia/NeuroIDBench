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
from deeb.pipelines.base import Basepipeline

class Siamese():
    def __init__(
        self,
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
        super().__init__(**kwargs)
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
        return ret
    
    # # This function has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons
    def _siamese_embeddings(self, no_channels, time_steps):
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
    
        

