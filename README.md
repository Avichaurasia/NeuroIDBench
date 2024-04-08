# NeuroIDBench

<!-- [![Pandas Latest Release](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.org/project/pandas/) [![MNE Latest Release - MNE](https://img.shields.io/pypi/v/mne.svg)](https://pypi.org/project/mne/) [![PyPI - Keras](https://img.shields.io/badge/Keras-latest-red.svg)](https://pypi.org/project/keras/) -->

<!-- <div align="center">
<img src="/images/brainModels.png" alt="My Logo" width="500" height="350">
</div>

</br> -->

This repository serves as a comprehensive resource for NeuroIDBench. It encompasses the entire implementation codebase along with a collection of illustrative examples for conducting benchmarking experiments using this powerful tool. Please note that while this repository is a valuable resource for code and methodologies, it does not include the proprietary or sensitive data utilized in our thesis.

The respository was intially created as part of the master thesis conducted by M.Sc [Avinash Kumar Chaurasia](https://avichaurasia.github.io/). It was written at the [IT Security](https://en.cs.uni-paderborn.de/its) group at Paderborn University, Germany under the supervision of Prof. Dr. [Patricia Arias Cabarcos](https://twitter.com/patriAriasC), who also leads the group. Further, the implementation aspects of this benchmarking tool was supervised by M.Sc [Matin Fallahi](https://ps.tm.kit.edu/english/21_318.php), a reserach associate at Kalrsruhe Insistute of Technology, Germany. 

Moreover, a reaearch paper was written as an extension of the master thesis. The paper was submitted to the [Journal of Information Security and Applications](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications) on 31st January 2024. While the paper undergoes review, Pre-Print of the paper can be found at [Arxiv](https://arxiv.org/abs/2402.08656).  

## Table of Contents

- [NeuroIDBench Architecture](#BrainModels-Architecture)
- [Installation](#installation)
- [Running](#Running)
- [Add new EEG data](NEWDATA.md)
- [Evaluate your own Twin Neural Network](RESEARCHERMETHOD.md)
- [Results](#results)
- [Cite our work](#cite-our-work)
- [References](#References)

## NeuroIDBench Architecture

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
git clone https://github.com/Avichaurasia/NeuroIDBench.git
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

## Results

### Single Session vs Multi Session Authentication

<b> Single Session Authentication</b>: In the single-session authentication, feature training and testing utilize recorded data from a singular session. This evaluation is conducted under both known and unknown attacker scenarios. In the known attacker scenario, attackers are assumed to be part of the system. Conversely, the unknown attacker scenario mirrors a more realistic environment, where attackers are external to the system. This presents a greater challenge for the model, as it must accurately identify attackers whose EEG data was not included during the training phase.

<!-- <div align="center">
<img src="/Plots/Single_Session_Evaluation/Single_Session_Evaluation.png" alt="Single Session Evaluation under known and unknown attacker scenarios" width="800" height="250">
</div> -->

<!-- ### Multi Session Evaluation -->

<b> Multi Session Authentication</b>: In multi-session evaluation, the training and testing of features extend across multiple sessions, encompassing data collected over various time periods or from different users. This evaluation method offers a comprehensive assessment of the system's performance over time and across different user cohorts. By incorporating data from multiple sessions, the model's ability to generalize across diverse conditions and adapt to evolving user patterns is evaluated. Multi-session evaluation is particularly valuable for assessing the long-term reliability and robustness of EEG-based authentication systems, as it accounts for variability in user behavior and data collection conditions across different sessions.

<div align="center">
<img src="/Plots/Evalaution.png" alt="Multi Session Evaluation unknown attacker scenarios" width="600" height="600">
</div>

<!-- The results of single single authentication reveal that the mean EER across datasets degraded by 58.44% for KNN, 275.60% for LDA, 383.91% for LR, 5.83% for NB, 75.94% for RF, 66.61% for SVM, and 75.88% for the Twin neural network approach. Notably, in some cases, the EER increased several-fold.

The results indicate a significant increase in EER compared to the single-session results. Notably, LDA and LR, which were not among the best perform-
ers in the single-session scenario, show more promising results in the multi-session context.  -->

### Pre-processing Parameters Impact

<b>Brain Sample Duration</b>: We investigated the impact of different sample durations, ranging from 1.0 seconds to 2.0 seconds in increments of 0.2 seconds, on the performance of multiple classifiers. Our analysis delved into how these varying sample durations influenced the effectiveness of the classifiers in achieving accurate predictions.

<div align="center">
<img src="/Plots/Experiment_Pre_Processing/Epochs_Interval/Epochs_Duration.png" alt="Multi Session Evaluation unknown attacker scenarios" width="800" height="380">
</div>



### Effects of Time and Frequency Domain Features

We also explored some of the most commonly used feature extraction methods in brainwave authentication such as Power Spectral Density (PSD) and Autoregressive (AR) models of different orders as feature extraction methods typically employed in shallow classifiers. Figure 6 illustrates the EER for various configurations of feature extraction across different classifiers. The findings suggest that the combination of PSD with AR of order 1 yields superior performance compared to other combinations. Following this, PSD features alone demonstrate promising results, whereas AR on its own fails to show stable and robust outcomes. Interestingly, in the BrainInvaders15a dataset, the AR of order 1 outperformed most classifiers. In the ERPCORE P300 dataset, PSD was the predominant feature leading to superior performance across most
classifiers. A

<div align="center">
<img src="/Plots/Experiment_feature_Extraction/Feature_Extraction.png" alt="Multi Session Evaluation unknown attacker scenarios" width="800" height="550">
</div>


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









