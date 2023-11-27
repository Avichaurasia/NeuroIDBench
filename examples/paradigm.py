import sys
sys.path.append('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
#from brainmodels.paradigms.base_old import BaseParadigm
#from deeb.paradigms.erp import N400
#from brainmodels.paradigms.n400 import N400
from brainModels.preprocessing.erp import ERP
from brainModels.datasets.brainInvaders15a import BrainInvaders2015a
from brainModels.datasets.mantegna2019 import Mantegna2019
#from brainmodels.datasets.draschkow2018 import Draschkow2018
from brainModels.datasets.won2022 import Won2022
from brainModels.featureExtraction.features import AutoRegressive as AR
from brainModels.featureExtraction.features import PowerSpectralDensity as PSD
#from deeb.pipelines.siamese_old import Siamese
from brainModels.featureExtraction.base import Basepipeline
#from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from brainModels.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from deeb.pipelines.siamese import Siamese
from brainModels.datasets.erpCoreN400 import ERPCOREN400

def _evaluate():
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
    #paradigm_p300=P300()
    paradigm_n400=ERP()

    # Getting the pre-prpcessed data for the p300 paradigm
    X, sub, meta=paradigm_n400.get_data(mantegna, return_epochs=True)
    print("length of X", len(X))


if __name__ == '__main__':
#    package = 'deeb.Evaluation'  # Change to your package/module name
#    for importer, modname, ispkg in pkgutil.walk_packages(path=['/Users/avinashkumarchaurasia/Desktop/deeb/deeb/Evaluation'], prefix=package + '.'):
#         print('Found submodule %s (is a package: %s)' % (modname, ispkg))
    print(sys.path)
    result= _evaluate()
    