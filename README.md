# BrainModels

[![Pandas Latest Release](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.org/project/pandas/) [![MNE Latest Release - MNE](https://img.shields.io/pypi/v/mne.svg)](https://pypi.org/project/mne/) [![PyPI - Keras](https://img.shields.io/badge/Keras-latest-red.svg)](https://pypi.org/project/keras/)

<div align="center">
<img src="/images/brainModels.png" alt="My Logo" width="500" height="350">
</div>

</br>

This repository serves as a comprehensive resource for BrainModels. It encompasses the entire implementation codebase along with a collection of illustrative examples for conducting benchmarking experiments using this powerful tool. Please note that while this repository is a valuable resource for code and methodologies, it does not include the proprietary or sensitive data utilized in our thesis.

The respository was intially created as part of the master thesis conducted by M.Sc [Avinash Kumar Chaurasia](https://avichaurasia.github.io/). It was written at the [IT Security](https://en.cs.uni-paderborn.de/its) group at Paderborn University, Germany under the supervision of Prof. Dr. [Patricia Arias Cabarcos](https://twitter.com/patriAriasC), who also leads the group. Further, the implementation aspects of this benchmarking tool was supervised by M.Sc [Matin Fallahi](https://ps.tm.kit.edu/english/21_318.php), a reserach associate at Kalrsruhe Insistute of Technology, Germany. 

Moreover, a reaearch paper was written as an extension of the master thesis. The paper was submitted to the [Journal of Information Security and Applications](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications) on 31st January 2024. While the paper undergoes review, Pre-Print of the paper can be found at [Arxiv](https://arxiv.org/abs/2402.08656).  

## Table of Contents

- [BrainModels Architecture](#BrainModels-Architecture)
- [Installation](#installation)
- [Running](#Running)
- [Add new EEG data](#Add-new-EEG-data)
- [Evaluate your own Twin Neural Network](#Evaluate-your-own-Twin-Neural-Network)
- [Cite our work](#cite-our-work)
- [References]{#References}

## BrainModels Architecture

<div align="center">
<img src="/images/Architecture.png" alt="Architecture" width="800" height="380">
</div>

There are four main concepts for this framework: datasets, PreProcessing, FeatureExtraction, Evalautions. Furthermore, 
we provide statistical and visualization tools to streamline the process.

### Datasets: 

This module offers abstract access to open datasets. It entails downloading open datasets from the internet and 
providing effective data management.

### PreProcesing: 

The purpose of this module is to conduct pre-processing on the unprocessed EEG data. 
Datasets exhibit distinct characteristics based on ERP paradigms such as P300 and N400. Nevertheless, both conditions 
elicit ERP responses after the individual’s exposure to unexpected stimuli. Consequently, the datasets for the P300 
and N400 paradigms undergo pre-processing using identical parameters.

### FeatureExtraction: 

This module extracts features from data that has been pre-processed. These characteristics are extracted in the time 
domain using Auto Regressive Coeffecients and in the frequency domain using Power Spectral Density. Furthermore, this module
also provides Siamese Network Architecture.  

### Evaluations: 

Evaluation defines the different authentication strategy which involves single session evalaution 
and multi-session evalaution under both known and unknown attack scenarios. 

### Analysis

Once an evaluation has been run, the raw results are returned as a DataFrame. The results such as ROC-Curve or EER can be 
visualized by calling functions from this module. 


## Installation

Get the project by cloning from github

```bash
git clone https://github.com/NeuroBench/neurobench.git
```

To run this project, several dependencies need to be installed. 


Install all these dependencies at once using the following command:

```bash
numpy 
scipy 
mne 
pandas 
scikit-learn 
matplotlib 
seaborn 
pooch 
requests 
tqdm 
zipfile36 
statsmodels 
mat73 
tensorflow 
tensorflow_addons
```

To install these dependencies, please use the following command:

```bash
conda env create -f BrainModels/environment.yml
```

we can also utilize requirement.txt to create the virtual environment by using the pip
command

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Activate the conda environment using the following command:

```bash
conda activate master_thesis
```

## Running 

The tool can be run by two ways:

1. Running the tool through automated creation of authentication pipelines

<!-- Some of the examples for configuring yaml for developing automated authentication pipelines
can be found [here](CONFIGURATION.md). -->
Upon activating the Conda environment, navigate to the designated project directory. 
A file named single_dataset.yml can be located within the "configuration_files" folder. 
The single_dataset.yml file is adjusted based on the [exemplified](CONFIGURATION.md) configurations.
The YAML configuration files can be utilized to perform automated benchmarking and reproduce the 
results for the single dataset.

2. Running the tool through jupyter notebooks

Examples of evaluating across various datasets and schemes can be found in our [Jupyter Notebook examples](./Jupypter_Notebooks/).

## Add new EEG data

Reserachers can utilize this tool to perform benchmarking and evalaute their approach. 
However,there are certain pre-requisities that need to be fuflfilled to add new EEG data to this tool.
Following steps needs to be followed to add new EEG data.

1. Convert the raw EEG data into standarized MNE raw object(mne.io.Raw). MNE is a powerful Python package and capable to reading and 
    converting any EEG format into MNE format. Some of the tutorials for converting EEG data into standarized MNE format
    can be found at https://mne.tools/stable/auto_tutorials/io/index.html. 

2. Once the raw EEG data is converted into mne. Save the state of MNE object into .fif format. 
    The mne data should be saved in hierarchy of folders like "sujectID"--> "Sesssion number" --> "Run number" --> EEG_data.fif.
    For example: Assume an EEG dataset comprises of 2 subjects. Each subject has performed EEG task across 2 sessions
    and each session contains two runs. First create a folder with name <b>User_data</b> on your local system. This folder should contain the MNE data of users. Then the mne data should be saved inside <b>User_data</b> folder in the following way: 

    <b>Subject 1</b>: 
    "Subject_1" --> "Session_1" --> "Run_1" --> EEG_data.fif, 
    "Subject_1" --> "Session_2" --> EEG_data.fif 

    <b>Subject</b>:
    "Subject_2" --> "Session_1" --> "Run_1" --> EEG_data.fif, 
    "Subject_2" --> "Session_2" --> "Run_1" --> EEG_data.fif 

3. Edit the single_dataset.yml with the below configurations:

Benchamrking pipeline for User i.e., Reseracher's own MNE data with traditional and deep learning methods

```bash
name: "User"

dataset: 
  - name: UserDataset
    from: brainModels.datasets
    parameters: 
      dataset_path: '<local_system_to_folder_User_data>'
    
pipelines:

  "AR+PSD+SVM": 
    - name: AutoRegressive
      from: brainModels.featureExtraction
      parameters: 
        order: 6
        
    - name: PowerSpectralDensity
      from: brainModels.featureExtraction
        
    - name: SVC
      from: sklearn.svm
      parameters: 
        kernel: 'rbf'
        class_weight: "balanced"
        probability: True

  "TNN": 
    - name : TwinNeuralNetwork
      from: brainModels.featureExtraction
      parameters: 
        EPOCHS: 10
        batch_size: 256
        verbose: 1
        workers: 1

  
  "AR+PSD+RF": 
  - name: AutoRegressive
    from: brainModels.featureExtraction
    parameters: 
      order: 6
    
  - name: PowerSpectralDensity
    from: brainModels.featureExtraction
      
  - name: RandomForestClassifier
  
    from: sklearn.ensemble
    parameters: 
        class_weight: "balanced"
```

This benchmarking pipeline reads the MNE data from the folder User_data and create a dataset instance. 
Afterwards, the pipeline consisiting of traditional algorithm such as SVM and  deep learning method 
like Siamese Neural Networks is made.  

4. Launch the python file run.py from terminal which has a main method and internally calls the automation script for benchmark.py 


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
    from: brainModels.datasets
    parameters: 
      subjects: 10
      interval: [-0.1, 0.9]
      rejection_threshold: 200


pipelines:

  "TNN": 
    - name : TwinNeuralNetwork
      from:  brainModels.featureExtraction
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

## Cite our work

```bash
@article{chaurasia2024neurobench,
  title={NeuroBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research},
  author={Chaurasia, Avinash Kumar and Fallahi, Matin and Strufe, Thorsten and Terh{\"o}rst, Philipp and Cabarcos, Patricia Arias},
  journal={arXiv preprint arXiv:2402.08656},
  year={2024}
}
```

## References

[1] V. Jayaram, A. Barachant, Moabb: trustworthy algorithm benchmarking for bcis, Journal of neural engineering 15 (6) (2018) 
    066011

[2] M. Fallahi, T. Strufe, P. Arias-Cabarcos, Brainnet: Improving brainwavebased biometric recognition with siamese networks, in: 2023 IEEE International Conference on Pervasive Computing and Communications (PerCom), IEEE, 2023, pp. 53–60

[3] Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues, P. L. C., Gautheret, V., & Congedo, M. (2019). Brain Invaders calibration-less P300-based BCI with modulation of flash duration Dataset (BI2015a) https://hal.archives-ouvertes.fr/hal-02172347

[4] Hinss, Marcel F., et al. "Open multi-session and multi-task EEG cognitive Dataset  for passive brain-computer Interface Applications." Scientific Data 10.1 (2023): 85.

[5] Kappenman, Emily S., et al. "ERP CORE: An open resource for human event-related potential  research." NeuroImage 225 (2021): 117465.

[6] Hübner, D., Verhoeven, T., Schmid, K., Müller, K. R., Tangermann, M., & Kindermans, P. J. (2017) Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees. PLOS ONE 12(4): e0175856.
https://doi.org/10.1371/journal.pone.0175856

[7] Mantegna, Francesco, et al. "Distinguishing integration and prediction accounts of ERP N400  modulations in language processing through experimental design." Neuropsychologia 134 (2019): 107199.

[8] Sosulski, J., Tangermann, M.: Electroencephalogram signals recorded from 13 healthy subjects during an auditory oddball paradigm under different stimulus onset asynchrony conditions. Dataset. DOI: 10.6094/UNIFR/154576

[9] K. Won, M. Kwon, M. Ahn, S. C. Jun, Eeg dataset for rsvp and p300 speller brain-computer interfaces, Scientific Data 9 (1) (2022) 388.

[10] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E., Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
toolbox for three BCI paradigms: An investigation into BCI illiteracy. GigaScience, 8(5), 1–16.
https://doi.org/10.1093/gigascience/giz002









