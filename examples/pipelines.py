import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base import BaseParadigm
from deeb.paradigms.p300 import N400
from deeb.paradigms.p300 import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive
from deeb.pipelines.features import PowerSpectralDensity
#from deeb.pipelines.siamese_old import Siamese
from deeb.pipelines.base import Basepipeline
#from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Intiaizing the datasets
dest = Won2022()
brain=Mantegna2019()

# Selecting the first 3 subjects from the Won2022 dataset
#dest.subject_list = dest.subject_list[0:5]

# Selecting the first 5 subjects from the Mantegna2019 dataset
brain.subject_list = brain.subject_list[0:5]

# Initializing the p300 paradigm
paradigm=P300()

# Getting the auto-regressive features from the Won2022 dataset for the p300 paradigm
ar=AutoRegressive(order=6)

# getting the power spectral density features from the Won2022 dataset for the p300 paradigm
psd=PowerSpectralDensity()

features=ar.get_data(dest, paradigm)
print("features", features)





