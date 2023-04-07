import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
import abc
import logging
import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base import BaseParadigm
from deeb.paradigms.erp import N400
from deeb.paradigms.erp import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipelines.features import AutoRegressive 
from deeb.pipelines.features import PowerSpectralDensity 
from deeb.pipelines.siamese_old import Siamese
from deeb.pipelines.base import Basepipeline
from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation
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
from deeb.analysis.plotting import Plots 
import os

# Function for performing evaulation across differeeent datasets and pipelines
def _evaluate():
    # Intiaizing the datasets
    #print(os.environ)
    won = Won2022()
    brain=BrainInvaders2015a()
    mantegna=Mantegna2019()

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
    pipeline['AR+SVM']=make_pipeline(AutoRegressive(order=10), SVC(kernel='rbf', probability=True))
    # pipeline['PSD+SVM']=make_pipeline(PowerSpectralDensity(), SVC(kernel='rbf', probability=True))
    pipeline['AR+LR']=make_pipeline(AutoRegressive(order=10), LogisticRegression())
    # pipeline['PSD+LR']=make_pipeline(PowerSpectralDensity(), LogisticRegression())
    pipeline['AR+LDA']=make_pipeline(AutoRegressive(order=10), LDA(solver='lsqr', shrinkage='auto'))
    # pipeline['PSD+LDA']=make_pipeline(PowerSpectralDensity(), LDA(solver='lsqr', shrinkage='auto'))
    pipeline['AR+NB']=make_pipeline(AutoRegressive(order=10), GaussianNB())
    # pipeline['PSD+NB']=make_pipeline(PowerSpectralDensity(), GaussianNB())
    pipeline['AR+KNN']=make_pipeline(AutoRegressive(order=10), KNeighborsClassifier(n_neighbors=3))
    # pipeline['PSD+KNN']=make_pipeline(PowerSpectralDensity(), KNeighborsClassifier(n_neighbors=3))
    pipeline['AR+RF']=make_pipeline(AutoRegressive(order=10), RandomForestClassifier(n_estimators=100))
    # # pipeline['PSD+RF']=make_pipeline(PowerSpectralDensity(), RandomForestClassifier(n_estimators=100))

    #pipeline['AR+NB']=make_pipeline(AutoRegressive(order=6), GaussianNB())
    #pipeline['AR+KNN']=make_pipeline(AutoRegressive(order=6), KNeighborsClassifier(n_neighbors=3))

    # Getting the results for the open set evaluation
    # evaluation=OpenSetEvaluation(paradigm=paradigm, datasets=dest, overwrite=False)

    # Getting the results for the close set evaluation
    open_set=CloseSetEvaluation(paradigm=paradigm_n400, datasets=mantegna, overwrite=False)
    results=open_set.process(pipeline)
    #print(os.environ)

    plot=Plots()
    plot._roc_curve_single_dataset(results, evaluation_type="Close-Set", dataset=mantegna)

    close_set=OpenSetEvaluation(paradigm=paradigm_n400, datasets=mantegna, overwrite=False)
    results_close_set=close_set.process(pipeline)
    #print(results_close_set['frr_1_far'])
    plot._roc_curve_single_dataset(results_close_set, evaluation_type="Open-Set", dataset=mantegna)
    # #print(datasets[0].dataset_path)
    return results


if __name__ == '__main__':
   result= _evaluate()
   print(result)
   #print(result)
#print(results['eer'])

