import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base import BaseParadigm
# from deeb.paradigms.p300 import N400
from deeb.paradigms.p300 import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive as AR
from deeb.pipelines.features import PowerSpectralDensity as PSD
from deeb.pipelines.base import Basepipeline
from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from deeb.datasets.lee2019 import Lee2019

# Intializing the datasets 
lee = Lee2019()
lee.subject_list = lee.subject_list[0:1]
print(lee.get_data()[1]['session_1']['train'].get_data().shape)

# brainInvaders=BrainInvaders2015a()
# mantegna=Mantegna2019()

# # Selecting the first 3 subjects from the Won2022 dataset
# won.subject_list = won.subject_list[0:10]

# # Selecting the first 5 subjects from the Mantegna2019 dataset
# brainInvaders.subject_list = brainInvaders.subject_list[0:5]

# # Downloading the dataset
# #dest.download()

# # getting the raw data for the dataset
# print(won.get_data())