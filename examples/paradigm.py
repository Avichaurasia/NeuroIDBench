import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/deeb')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base_old import BaseParadigm
#from deeb.paradigms.erp import N400
from deeb.paradigms.n400 import N400
from deeb.paradigms.p300 import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive as AR
from deeb.pipelines.features import PowerSpectralDensity as PSD
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
#from deeb.pipelines.siamese import Siamese
from deeb.datasets.erpCoreN400 import ERPCOREN400

# Intializing the datasets 
erpcore = ERPCOREN400()

# Intiaizing the datasets
won = Won2022()
brain=BrainInvaders2015a()
mantegna=Mantegna2019()

# # Selecting the first 3 subjects from the Won2022 dataset
# won.subject_list = won.subject_list[0:10]

# Selecting the first 5 subjects from the brainInvaders2015a dataset
#brain.subject_list = brain.subject_list[0:10]

# selecting the first 3 subjects from the Mantegna dataset
#mantegna.subject_list = mantegna.subject_list[0:10]

# Initializing the p300 paradigm
paradigm_p300=P300()
paradigm_n400=N400()

# Getting the pre-prpcessed data for the p300 paradigm
X, sub, meta=paradigm_n400.get_data(mantegna, return_epochs=True)
print("length of X", len(X))
# print("BrainInvaders", sub)

# for i, epochs in enumerate(X):
#     print("event id", epochs.event_id.values)

# #print(meta[['subject', 'event_id']].value_counts())
# target_index=meta[meta['event_id']=="Target"].index.tolist()

# print("lenght of target index", len(target_index))

# target_epochs=X[target_index]

# print("length of target epochs", len(target_epochs))

# print("subject labels", meta.iloc[target_index]["subject"].value_counts())



# # Getting the pre-prpcessed data for the p300 paradigm
# X, sub, meta=paradigm_p300.get_data(won, return_epochs=True)
# print("Won ", sub)


# getting the pre-processed data from n400 paradigm
# X, sub, meta=paradigm_n400.get_data(mantegna, return_epochs=True)
#print("length of X", len(X))

#print("length of meta", meta['subject'])
#print("Mantegna", sub)


