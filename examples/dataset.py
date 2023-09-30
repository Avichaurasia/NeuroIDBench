import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/deeb')
import abc
import logging
import mne
import numpy as np
import pandas as pd
#from deeb.paradigms.base_old import BaseParadigm
#from deeb.datasets.base import BaseDataset
# from deeb.paradigms.p300 import N400
#from deeb.paradigms.p300 import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
#from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive
from deeb.pipelines.features import PowerSpectralDensity
from deeb.pipelines.base import Basepipeline
#from deeb.Evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from deeb.datasets.lee2019 import Lee2019
from deeb.datasets.erpCoreN400 import ERPCOREN400
from deeb.paradigms.n400 import N400
from deeb.paradigms.p300 import P300
from deeb.datasets.cogBciFlanker import COGBCIFLANKER
#from deeb.datasets.userDataset import User
import importlib
#importlib.reload(N400)

# Intializing the datasets 
#lee = Lee2019()
erp=ERPCOREN400()
mantegna2019=Mantegna2019()
#mantegna.interval
#p300=P300()
n400=N400()
#user=User()

print(erp.get_data())
# erp.subject_list = erp.subject_list[0:5]
# erp.rejection_threshold=200e-6
# print("Rejection threshold:", erp.rejection_threshold)
# #print(dir(n400))
# data, subject_dict, _=n400.get_data(erp)
# print("Chaurasia")
# #print(subject_dict)
# print(data)
# for subject, sessions in data.items():
#     for session, runs in sessions.items():
#         for run, raw_events in runs.items():
#             raw = raw_events[0]
#             print("Subject:", subject)
#             print("Session:", session)
#             print("Run:", run)
#             print("Raw data:", raw)
#             print("---------------------")




#data, events=erp.get_data()
#print(data)
#print(erp.get_data())
#lee.subject_list=lee.subject_list[0:1]
#lee.subject_list = lee.subject_list[0:5]
#print(lee.get_data())

#lee.get_data()
#ar=AutoRegressive(order=6)


#n400=N400()

# _, sub, meta=n400.get_data(erpcore, return_epochs=True)
# print(sub)


#print(erpcore.get_data())
# data=erpcore.get_data()
# print(data)
#lee.subject_list = lee.subject_list[0:1]
#print(lee.get_data()[1]['session_1']['train'].get_data().shape)

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