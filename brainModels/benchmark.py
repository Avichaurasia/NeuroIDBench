import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
#print("Running benchmark")
from brainModels.preprocessing import ERP
from brainModels.featureExtraction import (
    parse_pipelines_from_directory,
    generate_paradigms,
    parse_pipelines_for_single_dataset,
    get_paradigm_from_config,)
import os
from copy import deepcopy
from glob import glob
import logging
import os.path as osp
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
from brainModels.evaluations import (SingleSessionCloseSet, SingleSessionOpenSet)
log = logging.getLogger(__name__)

def benchmark(subjects=None,
              pipelines="./configuration_files/",
              single_session_evaluations=None,
              multi_session_evaluations=None,
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

        - evaluations: List of evaluation types to perform. (Default: ['Single_Session_Close_Set', 'Single_Session_Open_Set'])

        - paradigms: List of paradigms to evaluate. (Default: None)

        - results: Path to the directory for storing results. (Default: './results')

        - output: Path to the directory for benchmarking output. (Default: "./benchmark/")

        - contexts: Path to a YAML configuration file for context parameters. (Default: None)

    Returns:
        - Pandas DataFrame containing the benchmark results.
    """

    if single_session_evaluations is None:
        single_session_evaluations=['Single_Session_Close_Set', 'Single_Session_Open_Set']

    if multi_session_evaluations is None:
        multi_session_evaluations=['Single_Session_Close_Set', 'Single_Session_Open_Set', 'multi_Session_Close_Set', 'multi_Session_Open_Set']

    eval_type={'Single_Session_Close_Set':SingleSessionCloseSet,
               'Single_Session_Open_Set':SingleSessionOpenSet}
    
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
    #print(type(ERP))
    p=ERP()
    log.debug(context_params['paradigm'])  
    if (dataset.n_sessions==1):
        evaluations=single_session_evaluations

    else:
        evaluations=multi_session_evaluations
        
    for eval in evaluations:
        ppl_with_epochs, ppl_with_array = {}, {} 
        for pn, pv in prdgms['pipelines'].items():
            ppl_with_epochs[pn]=pv 
            context = eval_type[eval](
                    paradigm=p,
                    datasets=dataset,
                )
        # Calling the evualtion function
        paradigm_results = context.process(pipelines=ppl_with_epochs)  
        df_eval.append(paradigm_results) 
    return pd.concat(df_eval)
