# DEEB

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

Execute the Automation Script

Launch the automated script single_dataset_benchmark.py
in Python. This script streamlines all the tasks related to data preprocessing, feature extraction,
and classification for a single dataset. It conducts benchmarking assessments across multiple
classifiers for the specified dataset

## How to Add new EEG data for benchmarking?

Reserachers can utilize this tool to perform benchmarking and evalaute their approach. 
However,there are certain pre-requisities that need to be fuflfilled to add new EEG data to this tool.
Following steps needs to be followed to add new EEG data.

1. Convert the raw EEG data into standarized MNE raw object. MNE is a powerful Python package and capable to reading and 
    converting any EEG format into MNE format. Some of the tutorials for converting EEG data into standarized MNE format
    can be found at https://mne.tools/stable/auto_tutorials/io/index.html. 

2. Once the raw EEG data is converted into mne. Save the state of MNE object into .fif format. 
    The mne data should be saved in hierarchy of folders like "sujectID"--> "Sesssion number" --> EEG_data.fif.
    For example: Assume an EEG dataset comprises of 2 subjects. Each subject has performed EEG task across 2 sessions
    and each session contains two runs. 

3. Edit the single_dataset.yml with the below configurations:

4. Launch the automated script single_dataset_benchmark.py from terminal.  









