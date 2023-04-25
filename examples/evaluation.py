import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/deeb')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base import BaseParadigm
from deeb.paradigms.n400 import N400
from deeb.paradigms.p300 import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive 
from deeb.pipelines.features import PowerSpectralDensity 
from deeb.pipelines.base import Basepipeline
from deeb.Evaluation.evaluation_old import CloseSetEvaluation, OpenSetEvaluation
from deeb.Evaluation.within_session_evaluation import WithinSessionEvaluation
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from deeb.datasets.erpCoreN400 import ERPCOREN400
from deeb.analysis.plotting import Plots 
import os

# Function for performing evaulation across differeeent datasets and pipelines
def _evaluate():
    # Intiaizing the datasets
    #print(os.environ)
    won = Won2022()
    brain=BrainInvaders2015a()
    mantegna=Mantegna2019()
    erp_core=ERPCOREN400()
    #erp_core.subject_list = erp_core.subject_list[0:5]

    # mantegna.subject_list = mantegna.subject_list[0:10]

    #Selecting the first 3 subjects from the Won2022 dataset
    # won.subject_list = won.subject_list[0:10]

    # # Selecting the first 5 subjects from the Mantegna2019 dataset
    # brain.subject_list = brain.subject_list[0:5]

    # Creating a list of datasets
    #datasets=[brain]

    # Initializing the p300 paradigm
    paradigm=P300()
    paradigm_n400=N400()

    # Intializing the pipelines
    pipeline={}
    pipeline['AR+PSD+SVM']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), SVC(kernel='rbf', probability=True))
    #pipeline['PSD+SVM']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), SVC(kernel='rbf', probability=True))
    pipeline['AR+PSD+LR']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LogisticRegression())
    # #pipeline['PSD+LR']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LogisticRegression())
    pipeline['AR+PSD+LDA']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LDA(solver='lsqr', shrinkage='auto'))
    # #pipeline['PSD+LDA']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), LDA(solver='lsqr', shrinkage='auto'))
    pipeline['AR+PSD+NB']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), GaussianNB())
    # #pipeline['PSD+NB']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), GaussianNB())
    pipeline['AR+PSD+KNN']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), KNeighborsClassifier(n_neighbors=3))
    #pipeline['PSD+KNN']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), KNeighborsClassifier(n_neighbors=3))
    pipeline['AR+PSD+RF']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), RandomForestClassifier())
    #pipeline['PSD+RF']=make_pipeline(AutoRegressive(order=6), PowerSpectralDensity(), RandomForestClassifier(n_estimators=100))

    #pipeline['AR+NB']=make_pipeline(AutoRegressive(order=6), GaussianNB())
    #pipeline['AR+KNN']=make_pipeline(AutoRegressive(order=6), KNeighborsClassifier(n_neighbors=3))

    # Getting the results for the open set evaluation
    # evaluation=OpenSetEvaluation(paradigm=paradigm, datasets=dest, overwrite=False)

    # # # Getting the results for the close set evaluation
    # open_set=CloseSetEvaluation(paradigm=paradigm_n400, datasets=erp_core, overwrite=False)
    # results=open_set.process(pipeline)
    # # # #print(os.environ)
    # plot=Plots()
    # plot._roc_curve_single_dataset(results, evaluation_type="Close-Set", dataset=erp_core)

    # close_set=OpenSetEvaluation(paradigm=paradigm_n400, datasets=erp_core, overwrite=False)
    # results_close_set=close_set.process(pipeline)
    # # # #print(results_close_set['frr_1_far'])
    # plot._roc_curve_single_dataset(results_close_set, evaluation_type="Open-Set", dataset=erp_core)
    # # # # #print(datasets[0].dataset_path)
    #return results

    # Getting the results for the within session evaluation
    within_session=WithinSessionEvaluation(paradigm=paradigm_n400, datasets=erp_core, overwrite=False)
    results_within_session=within_session.process(pipeline)
    return results_within_session


if __name__ == '__main__':
   result= _evaluate()
   print(result)
   #print(result)
#print(results['eer'])
