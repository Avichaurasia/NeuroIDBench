
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
#import tensorflow as tf
#from .features import AR, PSD

dest = Won2022()
#dest=[dset]
brain=Mantegna2019()
dest.subject_list = dest.subject_list[0:3]
brain.subject_list = brain.subject_list[0:5]

# Downloading the dataset
dest.download()

# getting the raw data for the dataset
print(dest.get_data())
# dest=[dest, brain]
# paradigm=P300()
# pipeline={}
# pipeline['AR+SVM']=make_pipeline(AR(), SVC(kernel='rbf', probability=True))
# pipeline['PSD+NB']=make_pipeline(PSD(), GaussianNB())
# pipeline['AR+LDA']=make_pipeline(AR(order=4), LDA(solver='lsqr', shrinkage='auto'))
# #evaluation=CloseSetEvaluation(paradigm=paradigm, datasets=dset, overwrite=False)
# evaluation=OpenSetEvaluation(paradigm=paradigm, datasets=dest, overwrite=False)
# results=evaluation.process(pipeline)
# print(results)

# for name, clf in pipeline.items():
#     #print(name)
#     #print(clf[0])
#     featiures=clf[0].get_data(dset, paradigm)
#     print(featiures)
#print(pipeline['AR+LDA']['autoregressive'])
#psd=PowerSpectralDensity()
#ar=AutoRegressive(order=4)
#print(dset.subject_list)
#snn=Siamese(optimizer='Adam')
#print(paradigm.used_events(dset))
#X, sub, meta=paradigm.get_data(dset, return_epochs=False, return_raws=False)
#print("X", sub)
#preds, threshold=snn.get_data(dset, paradigm, return_epochs=True, return_raws=False)
#print("Threshold", threshold)