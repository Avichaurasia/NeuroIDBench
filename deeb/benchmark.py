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
from deeb.evaluation import (CloseSetEvaluation, OpenSetEvaluation,)
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

    prdgms = generate_paradigms(pipeline_config, context_params, log)
    #print(prdgms)
    if paradigms is not None:
        prdgms = {p: prdgms[p] for p in paradigms}

    #print("Paradigms: ", prdgms)
    if len(context_params) == 0:
        #print("Avinash")
        for paradigm in prdgms:
            context_params[paradigm] = {}

    #print("Context: ", context_params)
    df_eval = []
    for evaluation in evaluations:
        eval_results = dict()
        for paradigm in prdgms:
            # get the context
            log.debug(f"{paradigm}: {context_params[paradigm]}")
            p = getattr(deeb_paradigms, paradigm)(**context_params[paradigm])
            # print("Paradigm: ", p)
            # List of dataset class instances


            ## Need to fix this
            datasets_for_paradigm = p.datasets

            # Selecting 10 subjects from each dataset and then return the updated datasets object
            datasets = []
            if(subjects is not None):
                for data in datasets_for_paradigm:
                    data.subject_list = data.subject_list[:subjects]
                    datasets.append(data)
            else:
                datasets = datasets_for_paradigm

            # print("BrainInvaders15a",datasets[0].subject_list)
            # print("Won2022",datasets[1].subject_list)
            # for df in datasets:
            #     print(df)
            #     print("10 subjects",df.subject_list)
            ppl_with_epochs, ppl_with_array = {}, {}
            for pn, pv in prdgms[paradigm].items():
                if "braindecode" in pn:
                    ppl_with_epochs[pn] = pv
                else:
                    ppl_with_array[pn] = pv

            #print("ppl_with_epochs", ppl_with_epochs)
            #print("p", p)
            #print("datasets", datasets)
            context = eval_type[evaluation](
                    paradigm=p,
                    datasets=datasets,
                    # random_state=42,
                    # hdf5_path=results,
                    # n_jobs=1,
                    # return_epochs=True,
                )
            
            # Calling the evualtion function
            paradigm_results = context.process(
                    pipelines=ppl_with_array
                )
            
            df_eval.append(paradigm_results)

        df_eval = pd.concat(df_eval)
        return df_eval
            

# Creating main function
if __name__ == "__main__":
    # Creating an object
    #brainInvaders=brainInvaders15a.BrainInvaders2015a()
    #paradigm_p300=P300()
    #obj = CloseSetEvaluation()
    # Calling function
    #obj.run()
    result=benchmark(subjects=5)
    print(result)


