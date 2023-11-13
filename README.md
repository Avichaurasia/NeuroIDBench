# BrainModels

<div align="center">
<img src="logo.png" alt="My Logo" width="400" height="380">
</div>


[![Pandas Latest Release](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.org/project/pandas/) [![MNE Latest Release - MNE](https://img.shields.io/pypi/v/mne.svg)](https://pypi.org/project/mne/) [![PyPI - Keras](https://img.shields.io/badge/Keras-latest-red.svg)](https://pypi.org/project/keras/)

<details>
<summary><strong>📘 Short Summary of Thesis</strong></summary>

Brainwaves present a compelling avenue for secure person authentication because they are inherently unobservable externally and capable of facilitating liveness detection. Harnessing brainwave’s unique and individualistic attributes, they have found extensive utility in various authentication applications. Nonetheless, the domain of brainwave authentication research has witnessed an upsurge in diverse experimental setups and the meticulous fine-tuning of parameters to optimize authentication methodologies. The substantial diversity in their methods poses a significant obstacle in assessing and measuring authentic research advancements. 

To address this multifaceted issue, this thesis introduces a versatile and robust benchmarking framework tailored explicitly for brainwave authentication systems. This framework draws upon the resources of four publicly accessible medical-grade brainwave datasets. It is worth mentioning that our study encompasses a substantial sample size consisting of 195 participants. The number of participants in our study is noteworthy, particularly when compared to the customary approach in brainwave authentication research, which typically involves a participant pool about one-fifth the size of our study.

Our extensive assessment encompassed a variety of state-of-the-art authentication algorithms, including Logistic Regression, Linear Discriminant Analysis, Support Vector Machine, Naive Bayes, K-Nearest Neighbours, Random Forest, and advanced deep learning methods like Siamese Neural Networks. Our evaluation approach incorporated both within-session (single-session) and cross-session (multi-session) analysis, covering threat cases like close-set (seen attacker) and open-set (unseen attacker) scenarios to ensure the tool’s versatility in different contexts. 

In within-session evaluation, our framework showcased outstanding performance for several classifiers, particularly Siamese Networks, which achieved an Equal Error Rate of 1.60% in the unseen attacker scenario. Additionally, our benchmarking framework’s adaptability is a notable asset, allowing researchers to tailor pre-processing, feature extraction, and authentication parameters to suit their specific requirements.
</details>

This repository serves as a comprehensive resource for my master's thesis, "Brainwave-Based User Authentication Models". It encompasses the entire implementation codebase along with a collection of illustrative examples for conducting benchmarking experiments using this powerful tool. Please note that while this repository is a valuable resource for code and methodologies, it does not include the proprietary or sensitive data utilized in our thesis.

The thesis was written at the IT Security group at Paderborn University. It was supervised by Patricia Arias Cabarcos, who also leads the group. Further, the implementation aspects of this benchmarking tool was supervised by Matin Fallahi, a reserach associate at Kalrsruhe Insistute of Technology, Germany.  

## Installation

### BrainModels

To run this project, several dependencies need to be installed. 


You can install all these dependencies at once using the following command:

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
conda env create -f DEEB/environment.yml
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

Edit the configuration File: 

Upon activating the Conda environment, navigate to the designated project directory. 
A file named single_dataset.yml can be located within the "configuration_files" folder. 
The single_dataset.yml file is adjusted based on the exemplified configurations
in the following sections.

### Example 1: 
Benchmarking pipeline using the dataset’s default parameters and auto-regressive features with SVM classification

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: BrainInvaders2015a
    from: brainModels.datasets

pipelines:

  "AR+PSD+SVM": 
    - name: AutoRegressive
      from: brainModels.pipelines

    - name: SVC
      from: sklearn.svm
      parameters: 
        kernel: 'rbf'
        class_weight: "balanced"
        probability: True
```

This benchmarking pipeline consists of default dataset's parameters which means EEG data of all the subjects are utlized in this pipeline.
Further, default epochs interval is set [-0.2, 0.8], rejection_threshold is None. AR features with default order 6 and SVM with kernel 'rbf',
constitutes the pipeline. 

### Example 2: 
Benchmarking pipeline by setting parmeters for the dataset, AR features with AR order 6 and algorithm SVM with kernel 'rbf'

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: BrainInvaders2015a
    from: brainModels.datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200

  pipelines: 

  "AR+SVM":
    - name: AutoRegressive 
      from: brainModels.pipelines 
      parameters:
        order: 5

    - name: SVC
      from: sklearn.svm 
      parameters:
        kernel: ’rbf’ 
        class_weight: "balanced" 
        probability: True
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. AR features with order 5 and SVM with kernel 'rbf' depicts the parameters for the feature extraction
and authentication algorithm. 

### Example 3: 
Benchamrking pipeline for dataset BrainInvaders15a with AR and PSD features with classifier SVM. 

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: BrainInvaders2015a
    from: brainModels.datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200


  pipelines: 
  
  "AR+SVM":
    - name: AutoRegressive 
      from: brainModels.pipelines 
      parameters:
        order: 5

    - name: PowerSpectralDensity 
      from: brainModels.pipelines

    - name: SVC
      from: sklearn.svm 
      parameters:
        kernel: ’rbf’ 
        class_weight: "balanced" 
        probability: True
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. AR and PSD features with SVM is utilized to form the sklearn pipeline. 

### Example 4: 
Benchamrking pipeline for dataset BrainInvaders15a with Siamese Networks

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: BrainInvaders2015a
    from: brainModels.datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200

  pipelines:

  "Siamese":
    - name : Siamese
    from: brainModels.pipelines 
    parameters:
        EPOCHS: 10 
        batch_size: 256 
        verbose: 1 
        workers: 1  
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. Here, Siamese neural Network pipeline with parameters such EPOCHS=10, batch_size=256 is set for 
training the neural network.

### Example 5: 
Benchamrking pipeline for dataset BrainInvaders15a with traditional and deep learning methods

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: BrainInvaders2015a
    from: brainModels.datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200

  pipelines:

   "AR+SVM":
    - name: AutoRegressive 
      from: brainModels.pipelines 
      parameters:
        order: 5

    - name: PowerSpectralDensity 
      from: brainModels.pipelines

    - name: SVC
      from: sklearn.svm 
      parameters:
        kernel: ’rbf’ 
        class_weight: "balanced" 
        probability: True

   "Siamese":
    - name : Siamese
    from: brainModels.pipelines 
    parameters:
        EPOCHS: 10 
        batch_size: 256 
        verbose: 1 
        workers: 1  
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. Here, the pipeline consisiting of traditional algorithm such as SVM and deep learning method like Siamese Neural Networks is made.  

Launch the automation Script: 

Launch the python file run.py
with command "python run.py". This python file will internally automation script benchmark.py. The  script parses the above configuration file and streamlines all the tasks related to data preprocessing, feature extraction, and classification for a single dataset. It conducts benchmarking assessments across multiple
classifiers for the specified dataset

# Benchmarking Architecture and Main Concepts

<div align="center">
<img src="Architecture.png" alt="Architecture" width="800" height="380">
</div>

There are four main concepts for this framework: datasets, paradigm, pipeline, Evalaution. Furthermore, 
we provide statistical and visualization tools to streamline the process.

## Datasets: 

This module offers abstract access to open datasets. It entails downloading open datasets from the internet and 
providing effective data management.

## Paradigm: 

The purpose of this module is to conduct pre-processing on the unprocessed EEG data. 
Datasets exhibit distinct characteristics based on ERP paradigms such as P300 and N400. Nevertheless, both conditions 
elicit ERP responses after the individual’s exposure to unexpected stimuli. Consequently, the datasets for the P300 
and N400 paradigms undergo pre-processing using identical parameters.

## Pipeline: 

This module extracts features from data that has been pre-processed. These characteristics are extracted in the time 
domain using Auto Regressive Coeffecients and in the frequency domain using Power Spectral Density. Furthermore, this module
also provides Siamese Network Architecture.  

## Evaluation: 

Evaluation defines the different authentication strategy which involves within-session(single session recordings) evalaution 
and cross-session (multi-session recordings) evalaution under both close-set and open-set attack scenarios. 
Results for within-session and cross-session are presented with metrics like EER, AUC, FRR at 1% FAR.

## Results and Visualization

Once an evaluation has been run, the raw results are returned as a DataFrame. The results such as ROC-Curve or EER can be 
visualized by calling functions from this module. 


# How to Add new EEG data for benchmarking?

Reserachers can utilize this tool to perform benchmarking and evalaute their approach. 
However,there are certain pre-requisities that need to be fuflfilled to add new EEG data to this tool.
Following steps needs to be followed to add new EEG data.

1. Convert the raw EEG data into standarized MNE raw object(mne.io.Raw). MNE is a powerful Python package and capable to reading and 
    converting any EEG format into MNE format. Some of the tutorials for converting EEG data into standarized MNE format
    can be found at https://mne.tools/stable/auto_tutorials/io/index.html. 

2. Once the raw EEG data is converted into mne. Save the state of MNE object into .fif format. 
    The mne data should be saved in hierarchy of folders like "sujectID"--> "Sesssion number" --> EEG_data.fif.
    For example: Assume an EEG dataset comprises of 2 subjects. Each subject has performed EEG task across 2 sessions
    and each session contains two runs. Then the mne data should be saved in the following ways: 

    <b>Subject 1</b>: 
    "Subject_1" --> "Session_1" --> EEG_data.fif, 
    "Subject_1" --> "Session_2" --> EEG_data.fif 

    <b>Subject</b>:
    "Subject_2" --> "Session_1" --> EEG_data.fif, 
    "Subject_2" --> "Session_2" --> EEG_data.fif 

3. Edit the single_dataset.yml with the below configurations:

Benchamrking pipeline for User i.e., Reseracher's own MNE data with traditional and deep learning methods

```bash
name: "User"

dataset: 
  - name: UserDataset
    from: brainModels.datasets
    parameters: 
      dataset_path: '/Users/avinashkumarchaurasia/mne_data/Matin/dataset'
    
pipelines:

  "AR+PSD+SVM": 
    - name: AutoRegressive
      from: brainModels.pipelines
      parameters: 
        order: 6
        
    - name: PowerSpectralDensity
      from: brainModels.pipelines
        
    - name: SVC
      from: sklearn.svm
      parameters: 
        kernel: 'rbf'
        class_weight: "balanced"
        probability: True

  "Siamese": 
    - name : Siamese
      from: brainModels.pipelines
      parameters: 
        EPOCHS: 10
        batch_size: 256
        verbose: 1
        workers: 1

  
  "AR+PSD+RF": 
  - name: AutoRegressive
    from: brainModels.pipelines
    parameters: 
      order: 6
    
  - name: PowerSpectralDensity
    from: brainModels.pipelines
      
  - name: RandomForestClassifier
  
    from: sklearn.ensemble
    parameters: 
        class_weight: "balanced"
```

This benchmarking pipeline reads the MNE data from the custom path "/Users/avinashkumarchaurasia/mne_data/Matin/dataset".
and create a dataset instance. Afterwards, the pipeline consisiting of traditional algorithm such as SVM and 
deep learning method like Siamese Neural Networks is made.  

4. Launch the python file run.py from terminal which has a main method and internally calls the automation script for benchmark.py 









