## Evaluate your own Twin Neural Network

Reserachers can also evaluate their own approach of Siamese Neural Network(SNN). This benchmarking tool
faciliates the reserachers to write their own customized SNN method in a python file, store it locally
on the machine. This tool imports the researcher method during the run time and trains and test the EEG
data with their SNN method. Following are some of the steps that need to be followed in order to integrate 
reserachers method.

1.  Create python file, imports all the necessary tensorflow packages, and then write a function which
    accepts two parameters. First parameter is "number of channels in EEG data" and second is 
    "time points". The function should be names "_siamese_embeddings" and returns the model which converts the
    high dimensional EEG data into compact brain embeddings. 
    Below is an example of siamese function that can be written in a .py file. 

```bash
import tensorflow as tf
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import (
    Input, Dense, Activation, Lambda, Reshape, BatchNormalization,
  LeakyReLU, Flatten, Dropout, Add,
  MaxPooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D)
from keras.models import Sequential, Model, load_model, save_model
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
import tensorflow_addons as tfa
import tensorflow_addons as tfa
from tensorflow_addons.losses import TripletSemiHardLoss
  
# # Make a function for siamese network embeddings with triplet loss function
def _siamese_embeddings(no_channels, time_steps):

  activef="selu"
  chn=no_channels
  sn=time_steps

  input = tf.keras.layers.Input((chn, sn, 1))
  x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(input)
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
  
```

This function is written inside a .py named "TNN.py" and stored locally anywhere on the machine.

2. Edit the single_dataset.yml with the below configurations:

Benchamrking pipeline for User i.e., Reseracher's own customized method for Siamese Neural Network

```bash
name: "ERPCORE400"

dataset: 
  - name: ERPCOREN400
    from: neuroIDBench.datasets
    parameters: 
      subjects: 10
      interval: [-0.1, 0.9]
      rejection_threshold: 200


pipelines:

  "TNN": 
    - name : TwinNeuralNetwork
      from:  neuroIDBench.featureExtraction
      parameters: 
        user_tnn_path: "<local_system_path_to_TNN.py>"
        EPOCHS: 10
        batch_size: 256
        verbose: 1
        workers: 1
```

In the above configuration, user_tnn_path is the path to python file containing the customized Siamse method.
This benchmarking pipeline performs benchmarking on EEG data of ERPCOREN400 with the researchers customized 
Siamese method. If the reseracher has its own EEG data, then they can follow the instructions mentioned in the above section
to add new EEG data and then can evaulate their EEG data on their own Siamese method.

4. Launch the python file run.py from terminal which has a main method and internally calls the automation script benchmark.py.