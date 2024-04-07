import sys
import abc
import logging
import mne
import numpy as np
import pandas as pd
#from deeb.paradigms.base_old import BaseParadigm
from neuroIDBench.preprocessing.erp import ERP
from neuroIDBench.datasets.brainInvaders15a import BrainInvaders2015a
from neuroIDBench.datasets.mantegna2019 import Mantegna2019
#from deeb.datasets.draschkow2018 import Draschkow2018
from neuroIDBench.datasets.won2022 import Won2022
from neuroIDBench.datasets.cogBciFlanker import COGBCIFLANKER
from neuroIDBench.featureExtraction.features import AutoRegressive as AR
from neuroIDBench.featureExtraction.features import PowerSpectralDensity
#from deeb.pipelines.siamese_old import Siamese
from neuroIDBench.featureExtraction.base import Basepipeline
#from deeb.evaluation.siamese_evaluation import Siamese_WithinSessionEvaluation
#from deeb.Evaluation.siamese_cross import Siamese_CrossSessionEvaluation
from neuroIDBench.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from neuroIDBench.featureExtraction.twinNeural import TwinNeuralNetwork
from neuroIDBench.evaluations.single_session_open_set import SingleSessionOpenSet

from neuroIDBench.analysis.plotting import Plots 
import os

def _evaluate():
    # Intiaizing the datasets
    won = Won2022()
    brain=BrainInvaders2015a()
    mantegna=Mantegna2019()
    cog=COGBCIFLANKER()

    #mantegna.subject_list=mantegna.subject_list[0:10]

    paradigm_n400=ERP()

    print(mantegna)
    print(paradigm_n400)

    # Intializing the pipelines
    pipeline={}
    pipeline['TNN']=make_pipeline(TwinNeuralNetwork())

    cross_session=SingleSessionOpenSet(paradigm=paradigm_n400, datasets=cog, return_close_set=False)
    
    grouped_df=cross_session.groupby(['eval Type','dataset','pipeline','session']).agg({
                'auc': 'mean',
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()

    return grouped_df


if __name__ == '__main__':
   result= _evaluate()
   print(result)


    




