import sys
import abc
import logging
import mne
import numpy as np
import pandas as pd
#from bra.datasets.
from neuroIDBench.datasets.lee2019 import Lee2019
from neuroIDBench.datasets.erpCoreN400 import ERPCOREN400
from neuroIDBench.datasets.erpCoreP300 import ERPCORENP300
from neuroIDBench.datasets.cogBciFlanker import COGBCIFLANKER
from neuroIDBench.preprocessing.erp import ERP
from neuroIDBench.featureExtraction.features import PowerSpectralDensity
from neuroIDBench.featureExtraction.features import AutoRegressive
from neuroIDBench.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from neuroIDBench.evaluations.single_session_close_set import SingleSessionCloseSet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC                                                                                 
from sklearn.ensemble import RandomForestClassifier
from neuroIDBench.analysis.plotting import Plots 
import os
import pkgutil


# Function for performing evaulation across differeeent datasets and pipelines
def _evaluate():
    # Intiaizing the datasets

    erpcore=ERPCORENP300()
    cogbci=COGBCIFLANKER()
    cogbci.subject_list=cogbci.subject_list[0:7]
    paradigm=ERP()
    print(cogbci.get_data())


if __name__ == '__main__':
   result= _evaluate()
   print(result)

