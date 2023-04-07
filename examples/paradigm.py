import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base import BaseParadigm
from deeb.paradigms.erp import N400
from deeb.paradigms.erp import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive as AR
from deeb.pipelines.features import PowerSpectralDensity as PSD
from deeb.pipelines.siamese_old import Siamese
from deeb.pipelines.base import Basepipeline
from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Intiaizing the datasets
won = Won2022()
brain=BrainInvaders2015a()
mantegna=Mantegna2019()

# # Selecting the first 3 subjects from the Won2022 dataset
# won.subject_list = won.subject_list[0:10]

# # Selecting the first 5 subjects from the brainInvaders2015a dataset
# brain.subject_list = brain.subject_list[0:10]

# # selecting the first 3 subjects from the Mantegna dataset
# mantegna.subject_list = mantegna.subject_list[0:10]

# Initializing the p300 paradigm
paradigm_p300=P300()
paradigm_n400=N400()

# Getting the pre-prpcessed data for the p300 paradigm
# X, sub, meta=paradigm_p300.get_data(brain, return_epochs=True)
# print("BrainInvaders", sub)

# Getting the pre-prpcessed data for the p300 paradigm
X, sub, meta=paradigm_p300.get_data(won, return_epochs=True)
print("Won ", sub)


# # getting the pre-processed data from n400 paradigm
# X, sub, meta=paradigm_n400.get_data(mantegna, return_epochs=True)
# print("Mantegna", sub)


