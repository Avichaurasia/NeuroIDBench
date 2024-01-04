import sys
sys.path.append('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from brainModels.datasets.lee2019 import Lee2019
from brainModels.datasets.erpCoreN400 import ERPCOREN400
from brainModels.datasets.erpCoreP300 import ERPCORENP300
from brainModels.preprocessing.erp import ERP
from brainModels.featureExtraction.features import PowerSpectralDensity
from brainModels.featureExtraction.features import AutoRegressive
from brainModels.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from brainModels.evaluations.single_session_close_set import SingleSessionCloseSet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC                                                                                 
from sklearn.ensemble import RandomForestClassifier
from brainModels.analysis.plotting import Plots 
import os
import pkgutil


# Function for performing evaulation across differeeent datasets and pipelines
def _evaluate():
    # Intiaizing the datasets

    erpcore=ERPCOREN400()
    erpcore.subject_list=erpcore.subject_list[0:1]
    paradigm=ERP()
    print(erpcore.get_data())


if __name__ == '__main__':
#    package = 'deeb.Evaluation'  # Change to your package/module name
#    for importer, modname, ispkg in pkgutil.walk_packages(path=['/Users/avinashkumarchaurasia/Desktop/deeb/deeb/Evaluation'], prefix=package + '.'):
#         print('Found submodule %s (is a package: %s)' % (modname, ispkg))
   result= _evaluate()
   print(result)

