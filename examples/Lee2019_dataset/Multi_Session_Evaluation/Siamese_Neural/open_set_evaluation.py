import sys
import abc
import logging
import mne
import numpy as np
import pandas as pd
from brainModels.datasets.lee2019 import Lee2019
from brainModels.preprocessing.erp import ERP
from brainModels.featureExtraction.twinNeural import TwinNeuralNetwork
from brainModels.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from brainModels.evaluations.multi_session_open_set import MultiSessionOpenSet
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
    paradigm=ERP()
    
    # Intializing the pipelines
    pipeline={}
    pipeline['siamese']=make_pipeline(TwinNeuralNetwork(batch_size=192, EPOCHS=100))
    evaluation=MultiSessionOpenSet(paradigm=paradigm, datasets=lee)
    results=evaluation.process(pipeline)
    
    fname='multi_session_siamese_open_set_res_dataframe.csv'
    results.to_csv(fname)
    grouped_df=results.groupby(['eval Type','dataset','pipeline']).agg({
                #'accuracy': 'mean',
                #'auc': 'mean',
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}',
                'frr_0.1_far': lambda x: f'{np.mean(x)*100:.3f}',
                "frr_0.01_far": lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()

    return grouped_df


if __name__ == '__main__':
   result= _evaluate()
   print(result)

