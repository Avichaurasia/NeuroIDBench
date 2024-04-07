import sys
import abc
import logging
import mne
import numpy as np
import pandas as pd
from neuroIDBench.preprocessing.erp import ERP
from neuroIDBench.datasets.brainInvaders15a import BrainInvaders2015a
from neuroIDBench.datasets.mantegna2019 import Mantegna2019
from neuroIDBench.datasets.cogBciFlanker import COGBCIFLANKER
from neuroIDBench.datasets.lee2019 import Lee2019
#from brainmodels.datasets.draschkow2018 import Draschkow2018
from neuroIDBench.datasets.won2022 import Won2022
from neuroIDBench.featureExtraction.features import AutoRegressive 
from neuroIDBench.featureExtraction.features import PowerSpectralDensity 
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from neuroIDBench.evaluations.single_session_close_set import SingleSessionCloseSet
from neuroIDBench.evaluations.single_session_open_set import SingleSessionOpenSet
#from brainModels.evaluations.multi_session_open_set import MultiSessionOpenSet
#from brainModels.evaluations.multi_session_open_set_copy import MultiSessionOpenSet
from neuroIDBench.featureExtraction.twinNeural import TwinNeuralNetwork
from neuroIDBench.datasets.lee2019 import Lee2019
import os

def _evaluate():

    duration_list=[]
    for interval in [0.8, 1, 1.2, 1.4, 1.6, 1.8]:
    #for interval in [0.8]:
        data=Lee2019()
        data.interval=[-0.2, interval]
        paradigm_n400=ERP()

        pipeline={}
        pipeline['TNN']=make_pipeline(TwinNeuralNetwork(batch_size=192, EPOCHS=100))
        open_set=SingleSessionOpenSet(paradigm=paradigm_n400, datasets=data, overwrite=False)
        results=open_set.process(pipeline)

        duration=interval+0.2
        results['epochs_duration']=duration
        duration_list.append(results)

    df=pd.concat(duration_list, axis=0)
    curent_dir=os.getcwd()
    fname='epochs_duration_siamese_results.csv'
    results_directory=os.path.join(curent_dir, "Siamese_Results", fname)
    df.to_csv(fname)

if __name__ == '__main__':

    result= _evaluate()
    