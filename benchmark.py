import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
# from deeb.paradigms.erp import P300
# from deeb.paradigms.erp import N400
#from brainModels.paradigms.p300 import P300
#from brainModels.paradigms.n400 import N400
from brainModels.paradigms.erp import ERP
from brainModels.pipelines.utils import (
    parse_pipelines_from_directory,
    generate_paradigms,
    parse_pipelines_for_single_dataset,
    get_paradigm_from_config,)
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
from brainModels.datasets import (BrainInvaders2015a, Mantegna2019, ERPCOREN400, Lee2019, utils)
from brainModels.Evaluation import (WithinSessionEvaluation, CrossSessionEvaluation, 
                             Siamese_WithinSessionEvaluation, Siamese_CrossSessionEvaluation)
log = logging.getLogger(__name__)

def benchmark(subjects=None,
             #pipelines="../single_dataset_pipelines/",
              pipelines="single_dataset_pipelines",
              evaluations=None,
              paradigms=None,
              results='./results',
              output="./benchmark/",
              contexts=None,):
    """
    Benchmark a set of pipelines on a given dataset and evaluation.

    This function conducts benchmarking of pipelines on a specified paradigm and evaluation type. 
    It supports different evaluation scenarios, such as within-session and cross-session, and various paradigms.

    Parameters:
    - subjects: List of subjects to be included in the benchmark. (Default: None)

    - pipelines: Path to the pipeline configuration files. (Default: "single_dataset_pipelines")

    - evaluations: List of evaluation types to perform. (Default: ['Within_Session', 'Cross_Session', 'Siamese_WithinSession', 'Siamese_CrossSession'])

    - paradigms: List of paradigms to evaluate. (Default: None)

    - results: Path to the directory for storing results. (Default: './results')

    - output: Path to the directory for benchmarking output. (Default: "./benchmark/")

    - contexts: Path to a YAML configuration file for context parameters. (Default: None)

    Returns:
    - Pandas DataFrame containing the benchmark results.
    """

    if evaluations is None:
        evaluations = ['Within_Session', 'Cross_Session', 'Siamese_WithinSession', 'Siamese_CrossSession']

    #print("Current path", os.getcwd())

    eval_type={'Within_Session':WithinSessionEvaluation,
               'Cross_Session':CrossSessionEvaluation,
               'Siamese_WithinSession':Siamese_WithinSessionEvaluation,
               'Siamese_CrossSession':Siamese_CrossSessionEvaluation}
    
    output = Path(output)
    if not osp.isdir(output):
        os.makedirs(output)

    pipeline_config = parse_pipelines_for_single_dataset(pipelines)
    context_params = {}
    if contexts is not None:
        with open(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)

    prdgms = get_paradigm_from_config(pipeline_config, context_params, log)
    context_params["paradigm"] = {}
    df_eval = []
    dataset=prdgms['dataset']
    p=ERP()
    log.debug(context_params['paradigm'])  
    for pn, pv in prdgms['pipelines'].items():
        ppl_with_epochs, ppl_with_array = {}, {}  
        if "Siamese" in pn:
            ppl_with_epochs[pn]=pv
            #ppl_with_epochs[pn] = pv
            if (dataset.n_sessions>2):
                context = eval_type["Siamese_CrossSession"](
                        paradigm=p,
                        datasets=dataset,
                    )
            
                # Calling the evualtion function
                paradigm_results = context.process(
                    pipelines=ppl_with_epochs)   
                df_eval.append(paradigm_results)
            
            context = eval_type["Siamese_WithinSession"](
                        paradigm=p,
                        datasets=dataset,
                    )
            
            # Calling the evualtion function
            paradigm_results = context.process(
                pipelines=ppl_with_epochs)   
            df_eval.append(paradigm_results)

        else:
            ppl_with_array[pn] = pv
            if (dataset.n_sessions>2):
                context = eval_type["Cross_Session"](
                        paradigm=p,
                        datasets=dataset,
                    )
            
                # Calling the evualtion function
                paradigm_results = context.process(
                    pipelines=ppl_with_array)   
                df_eval.append(paradigm_results)
                    
            context = eval_type["Within_Session"](
                        paradigm=p,
                        datasets=dataset,
                    )
            
            # Calling the evualtion function
            paradigm_results = context.process(
                pipelines=ppl_with_array)   
            df_eval.append(paradigm_results)
         
    return pd.concat(df_eval)




