import sys
sys.path.append("./")
import abc
import logging
import mne
import numpy as np
import pandas as pd
from brainModels.datasets.lee2019 import Lee2019
from brainModels.datasets.erpCoreN400 import ERPCOREN400
from brainModels.preprocessing.erp import ERP
from brainModels.featureExtraction.twinNeural import TwinNeuralNetwork
from brainModels.datasets import utils
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

    lee=Lee2019()
    erpcore=ERPCOREN400
    paradigm=ERP()

    # Intializing the pipelines
    pipeline={}
    pipeline['TNN']=make_pipeline(TwinNeuralNetwork(batch_size=192, EPOCHS=100, learning_rate=0.0001))
    
    evaluation=SingleSessionCloseSet(paradigm=paradigm, datasets=erpcore)
    results=evaluation.process(pipeline)

    grouped_df=results.groupby(['eval Type','dataset','pipeline']).agg({
                #'accuracy': 'mean',
                #'auc': 'mean',
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()

    return grouped_df


if __name__ == '__main__':
#    package = 'deeb.Evaluation'  # Change to your package/module name
#    for importer, modname, ispkg in pkgutil.walk_packages(path=['/Users/avinashkumarchaurasia/Desktop/deeb/deeb/Evaluation'], prefix=package + '.'):
#         print('Found submodule %s (is a package: %s)' % (modname, ispkg))
   result= _evaluate()
   print(result)

