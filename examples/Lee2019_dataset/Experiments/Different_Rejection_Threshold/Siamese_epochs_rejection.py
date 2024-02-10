import sys
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
from brainModels.datasets.cogBciFlanker import COGBCIFLANKER
from brainModels.datasets.lee2019 import Lee2019
#from brainmodels.datasets.draschkow2018 import Draschkow2018
from brainModels.datasets.won2022 import Won2022
from brainModels.featureExtraction.features import AutoRegressive 
from brainModels.featureExtraction.features import PowerSpectralDensity 
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from brainModels.evaluations.single_session_close_set import SingleSessionCloseSet
from brainModels.evaluations.single_session_open_set import SingleSessionOpenSet
from brainModels.featureExtraction.twinNeural import TwinNeuralNetwork
from brainModels.datasets.lee2019 import Lee2019
import os
import gc

def _evaluate():
    # Intializing the datasets 
    rejection_threshold_list=[]
    for threshold in [None, 100, 150, 200, 250, 300, 350, 400]:
        data=Lee2019()
        data.rejection_threshold=threshold
        # Initializing the p300 paradigm
        paradigm_n400=ERP()
        pipeline={}
        pipeline['siamese']=make_pipeline(TwinNeuralNetwork(batch_size=192, EPOCHS=100))
        open_set=SingleSessionOpenSet(paradigm=paradigm_n400, datasets=data, overwrite=False)
        results=open_set.process(pipeline)

        if (threshold==None):
            thersh=0
        else:
            thersh=threshold
        results['Rejection_Threshold']=thersh

        rejection_threshold_list.append(results)
        del data, paradigm_n400, pipeline, open_set, results
        gc.collect()

    df=pd.concat(rejection_threshold_list, axis=0)
    curent_dir=os.getcwd()
    fname='epochs_rejection_siamese_results.csv'
    results_directory=os.path.join(curent_dir, "Siamese_Results", fname)
    df.to_csv(fname)


if __name__ == '__main__':

    result= _evaluate()
    