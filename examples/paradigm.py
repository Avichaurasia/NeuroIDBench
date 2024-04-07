import sys
import abc
import logging
import mne
import numpy as np
import pandas as pd
from neuroIDBench.preprocessing.erp import ERP
from neuroIDBench.datasets.brainInvaders15a import BrainInvaders2015a
from neuroIDBench.datasets.mantegna2019 import Mantegna2019
#from brainmodels.datasets.draschkow2018 import Draschkow2018
from neuroIDBench.datasets.won2022 import Won2022
from neuroIDBench.featureExtraction.features import AutoRegressive as AR
from neuroIDBench.featureExtraction.features import PowerSpectralDensity as PSD
#from deeb.pipelines.siamese_old import Siamese
from neuroIDBench.featureExtraction.base import Basepipeline
#from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
from neuroIDBench.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from deeb.pipelines.siamese import Siamese
from neuroIDBench.datasets.erpCoreN400 import ERPCOREN400

def _evaluate():
    # Intializing the datasets 
    erpcore = ERPCOREN400()

    # Intiaizing the datasets
    won = Won2022()
    brain=BrainInvaders2015a()
    mantegna=Mantegna2019()

    # Initializing the p300 paradigm
    paradigm_n400=ERP()

    # Getting the pre-prpcessed data for the p300 paradigm
    X, sub, meta=paradigm_n400.get_data(mantegna, return_epochs=True)
    print("length of X", len(X))


if __name__ == '__main__':
    print(sys.path)
    result= _evaluate()
    