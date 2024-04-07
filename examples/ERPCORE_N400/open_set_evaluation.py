import sys
import abc
import logging
import mne
import numpy as np
import pandas as pd
from neuroIDBench.datasets.lee2019 import Lee2019
from neuroIDBench.datasets.erpCoreN400 import ERPCOREN400
from neuroIDBench.datasets.erpCoreP300 import ERPCORENP300
from neuroIDBench.preprocessing.erp import ERP
from neuroIDBench.featureExtraction.twinNeural import TwinNeuralNetwork
from neuroIDBench.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from neuroIDBench.evaluations.single_session_open_set import SingleSessionOpenSet
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

    lee=Lee2019()
    erpcore=ERPCORENP300
    paradigm=ERP()
    
    # Intializing the pipelines
    pipeline={}
    pipeline['siamese']=make_pipeline(TwinNeuralNetwork(batch_size=192, EPOCHS=100))
    evaluation=SingleSessionOpenSet(paradigm=paradigm, datasets=erpcore, overwrite=False)
    results=evaluation.process(pipeline)

    grouped_df=results.groupby(['eval Type','dataset','pipeline']).agg({
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()

    return grouped_df


if __name__ == '__main__':
   result= _evaluate()
   print(result)

