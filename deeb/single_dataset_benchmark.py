import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


from deeb.datasets import brainInvaders15a, mantegna2019, utils
# from deeb.evaluation.evaluation import CloseSetEvaluation
# from deeb.paradigms.erp import P300
# from deeb.paradigms.erp import N400
from deeb.pipelines.utils import (
    parse_pipelines_from_directory,
    generate_paradigms,

)
import os
from collections import OrderedDict
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from glob import glob
import importlib

import logging
import os.path as osp
from pathlib import Path
import mne
import numpy as np
import yaml
import pandas as pd


from deeb import paradigms as deeb_paradigms
from deeb.Evaluation import (CloseSetEvaluation, OpenSetEvaluation,)
log = logging.getLogger(__name__)

def benchmark(subjects=None,
              pipelines="../pipelines",
              evaluations=None,
              paradigms=None,
              results='./results',
              output="./benchmark/",
              contexts=None,):
    """ Benchmark a set of pipelines on a given paradigm and evaluation"""
    if evaluations is None:
        evaluations = ['close_set','open_set']

    eval_type={'close_set':CloseSetEvaluation,
               'open_set':OpenSetEvaluation}
    
    output = Path(output)
    if not osp.isdir(output):
        os.makedirs(output)

    pipeline_config = parse_pipelines_from_directory(pipelines)

    context_params = {}
    if contexts is not None:
        print("changing context")
        with open(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)
            print("context_params",context_params)

