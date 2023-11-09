import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/deeb')
import abc
import logging
import mne
import numpy as np
import pandas as pd
#from deeb.paradigms.base_old import BaseParadigm
from brainModels.paradigms.n400 import N400
from brainModels.paradigms.p300 import P300
from brainModels.datasets.brainInvaders15a import BrainInvaders2015a
from brainModels.datasets.mantegna2019 import Mantegna2019
#from deeb.datasets.draschkow2018 import Draschkow2018
from brainModels.datasets.won2022 import Won2022
from brainModels.datasets.cogBciFlanker import COGBCIFLANKER
from brainModels.pipelines.features import AutoRegressive 
from brainModels.pipelines.features import PowerSpectralDensity 
#from deeb.pipelines.siamese_old import Siamese
from brainModels.pipelines.base import Basepipeline
#from deeb.evaluation.siamese_evaluation import Siamese_WithinSessionEvaluation
#from deeb.Evaluation.siamese_cross import Siamese_CrossSessionEvaluation
from brainModels.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from brainModels.pipelines.siamese import Siamese

from brainModels.analysis.plotting import Plots 
import os

def _evaluate():
    # Intiaizing the datasets
    #print(os.environ)
    won = Won2022()
    brain=BrainInvaders2015a()
    mantegna=Mantegna2019()
    cog=COGBCIFLANKER()

    #mantegna.subject_list=mantegna.subject_list[0:10]

    paradigm_n400=N400()

    print(mantegna)
    print(paradigm_n400)

    # Intializing the pipelines
    pipeline={}
    pipeline['siamese']=make_pipeline(Siamese())

    cross_session=Siamese_CrossSessionEvaluation(paradigm=paradigm_n400, datasets=cog, return_close_set=False)
    
    grouped_df=cross_session.groupby(['eval Type','dataset','pipeline','session']).agg({
                'accuracy': 'mean',
                'auc': 'mean',
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()

    return grouped_df


if __name__ == '__main__':
   result= _evaluate()
   print(result)


    




