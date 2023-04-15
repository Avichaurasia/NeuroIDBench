import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
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
from deeb.pipelines.siamese_old import Siamese
from deeb.pipelines.base import Basepipeline
from deeb.evaluation.siamese_evaluation import Siamese_WithinSessionEvaluation
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from deeb.pipelines.siamese import Siamese

from deeb.analysis.plotting import Plots 
import os

def _evaluate():
    # Intiaizing the datasets
    #print(os.environ)
    won = Won2022()
    brain=BrainInvaders2015a()
    mantegna=Mantegna2019()

    mantegna.subject_list=mantegna.subject_list[0:10]

    paradigm_n400=N400()

    print(mantegna)
    print(paradigm_n400)

    # Intializing the pipelines
    pipeline={}
    pipeline['siamese']=make_pipeline(Siamese())

    within_close_set=Siamese_WithinSessionEvaluation(paradigm=paradigm_n400, datasets=mantegna, return_open_set=False)
    
    results_close_set=within_close_set.process(pipelines=pipeline)


if __name__ == '__main__':
   result= _evaluate()
   print(result)


    




