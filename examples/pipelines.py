import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/deeb/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
#from deeb.paradigms.base_old import BaseParadigm
#from deeb.paradigms.p300 import N400
from brainModels.paradigms.p300 import P300
from brainModels.paradigms.n400 import N400
from brainModels.datasets.brainInvaders15a import BrainInvaders2015a
from brainModels.datasets.mantegna2019 import Mantegna2019
from brainModels.datasets.erpCoreN400 import ERPCOREN400
#from deeb.datasets.lee2019 import Lee2019
#from deeb.datasets.draschkow2018 import Draschkow2018
from brainModels.datasets.won2022 import Won2022
from brainModels.pipelines.features import AutoRegressive
from brainModels.pipelines.features import PowerSpectralDensity
from brainModels.pipelines.siamese import Siamese
#from deeb.pipelines.siamese_old import Siamese
from brainModels.pipelines.base import Basepipeline
#from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from brainModels.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Intiaizing the datasets
dest = Won2022()
brain=Mantegna2019()
erpcore=ERPCOREN400()
#lee=Lee2019()
erpcore.subject_list=erpcore.subject_list[0:5]
#lee.subject_list=lee.subject_list[0:3]

# Selecting the first 3 subjects from the Won2022 dataset
#dest.subject_list = dest.subject_list[0:5]

# Selecting the first 5 subjects from the Mantegna2019 dataset
#brain.subject_list = brain.subject_list[0:5]

# Initializing the p300 paradigm
paradigm=P300()
paradigm_n400=N400()

# Getting the auto-regressive features from the Won2022 dataset for the p300 paradigm
#ar=AutoRegressive(order=6)

# getting the power spectral density features from the Won2022 dataset for the p300 paradigm
#psd=PowerSpectralDensity()

# Initializing the siamese pipeline
siamese=Siamese()

features=siamese.get_data(lee, paradigm)
print("features", features)





