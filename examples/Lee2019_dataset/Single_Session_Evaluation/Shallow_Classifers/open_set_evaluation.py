import sys
sys.path.append('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from brainModels.datasets.lee2019 import Lee2019
from brainModels.preprocessing.erp import ERP
from brainModels.featureExtraction.features import PowerSpectralDensity
from brainModels.featureExtraction.features import AutoRegressive
from brainModels.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from brainModels.evaluations.single_session_open_set import SingleSessionOpenSet
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
    #lee.rejection_threshold=200e-6
    #lee.baseline_correction=False
    #lee.subject_list=lee.subject_list[0:10]
    
    # Intializing the pipelines
    pipeline={}
    pipeline['AR+SVM']=make_pipeline(AutoRegressive(order=6), SVC(kernel='rbf', probability=True))
    # pipeline['AR+PSD+LR']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LogisticRegression())
    # # #pipeline['PSD+LR']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LogisticRegression())
    # pipeline['AR+PSD+LDA']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LDA(solver='lsqr', shrinkage='auto'))
    #pipeline['siamese']=make_pipeline(Siamese())
    # #pipeline['PSD+LDA']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LDA(solver='lsqr', shrinkage='auto'))
    #pipeline['AR+PSD+NB']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), GaussianNB())
    # #pipeline['PSD+NB']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), GaussianNB())
    # pipeline['AR+PSD+KNN']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), KNeighborsClassifier(n_neighbors=3))
    # #pipeline['PSD+KNN']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), KNeighborsClassifier(n_neighbors=3))
    # pipeline['AR+PSD+RF']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), RandomForestClassifier())
    #pipeline['PSD+RF']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), RandomForestClassifier(n_estimators=100))
 
    evaluation=SingleSessionOpenSet(paradigm=paradigm, datasets=lee)
    results=evaluation.process(pipeline)

    grouped_df=results.groupby(['eval Type','dataset','pipeline']).agg({
                'accuracy': 'mean',
                'auc': 'mean',
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
