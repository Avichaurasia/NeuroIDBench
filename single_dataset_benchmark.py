#import sys
#sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/deeb')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
# from deeb.paradigms.erp import P300
# from deeb.paradigms.erp import N400
from deeb.paradigms.p300 import P300
from deeb.paradigms.n400 import N400
from deeb.pipelines.utils import (
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
from deeb.datasets import (BrainInvaders2015a, Mantegna2019, ERPCOREN400, Lee2019, utils)
#from deeb.datasets import brainInvaders15a, mantegna2019, erpCoreN400, lee2019, utils
#from deeb import paradigms as deeb_paradigms
#from deeb.Evaluation import (CloseSetEvaluation, OpenSetEvaluation,)
from deeb.Evaluation import (WithinSessionEvaluation, CrossSessionEvaluation, 
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
    """ Benchmark a set of pipelines on a given paradigm and evaluation"""
    if evaluations is None:
        evaluations = ['Within_Session', 'Cross_Session', 'Siamese_WithinSession', 'Siamese_CrossSession']

    print("Current path", os.getcwd())

    eval_type={'Within_Session':WithinSessionEvaluation,
               'Cross_Session':CrossSessionEvaluation,
               'Siamese_WithinSession':Siamese_WithinSessionEvaluation,
               'Siamese_CrossSession':Siamese_CrossSessionEvaluation}
    
    output = Path(output)
    if not osp.isdir(output):
        os.makedirs(output)

    pipeline_config = parse_pipelines_for_single_dataset(pipelines)
    #print("pipeline_config",pipeline_config[0]['pipeline']['autoregressive'].order)

    context_params = {}
    #print("contexts",contexts)
    if contexts is not None:
        #print("changing context")
        with open(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)
            #print("context_params",context_params)
    # prdgm = get_paradigm(pipeline_config, context_params, log)

    # print("Paradigms: ", prdgm)
    prdgms = get_paradigm_from_config(pipeline_config, context_params, log)
    print("Paradigms: ", prdgms['pipelines'].keys())
    #print("Paradigms: ", prdgms)
    #if len(context_params) == 0:
        #for paradigm in prdgm:
    context_params["paradigm"] = {}

    #print("Context: ", context_params)
    df_eval = []
    dataset=prdgms['dataset']

    if dataset.paradigm == "p300":
        #paradigm_300= P300()
        #p = getattr(paradigm_300)(**context_params['paradigm'])
        p=P300()
    else: 
        #paradigm_N400= N400()
        #p = getattr(paradigm_N400)(**context_params['paradigm'])
        p=N400()
    log.debug(context_params['paradigm'])  
    for pn, pv in prdgms['pipelines'].items():
        ppl_with_epochs, ppl_with_array = {}, {}  
        #print(pv)
        if "SNN" in pn:
            ppl_with_epochs[pn]=pv
            #ppl_with_epochs[pn] = pv
            if (dataset.n_sessions>2):
                context = eval_type["Siamese_CrossSession"](
                        paradigm=p,
                        datasets=dataset,
                        # random_state=42,
                        # hdf5_path=results,
                        # n_jobs=1,
                        # return_epochs=True,
                    )
            
                # Calling the evualtion function
                paradigm_results = context.process(
                    pipelines=ppl_with_epochs)   
                df_eval.append(paradigm_results)
            
            context = eval_type["Siamese_WithinSession"](
                        paradigm=p,
                        datasets=dataset,
                        # random_state=42,
                        # hdf5_path=results,
                        # n_jobs=1,
                        # return_epochs=True,
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
                        # random_state=42,
                        # hdf5_path=results,
                        # n_jobs=1,
                        # return_epochs=True,
                    )
            
                # Calling the evualtion function
                paradigm_results = context.process(
                    pipelines=ppl_with_array)   
                df_eval.append(paradigm_results)
                    
            context = eval_type["Within_Session"](
                        paradigm=p,
                        datasets=dataset,
                        # random_state=42,
                        # hdf5_path=results,
                        # n_jobs=1,
                        # return_epochs=True,
                    )
            
            # Calling the evualtion function
            paradigm_results = context.process(
                pipelines=ppl_with_array)   
            df_eval.append(paradigm_results)
         
    return pd.concat(df_eval)

    
    

# Creating main function
if __name__ == "__main__":
    # Creating an object

    # Calling function
    #obj.run()
    print("Current path", os.getcwd())
    result=benchmark()

    #print(result.columns)
    #result['pipeline']=result['pipeline'].apply(lambda x: x.split('+')[-1])
    grouped_df=result.groupby(['evaluation','pipeline', 'eval Type','dataset']).agg({
                "subject": 'nunique',
                #'n_samples': 'first',
                #'accuracy': 'mean',
                'auc': 'mean',
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()
    grouped_df.rename(columns={'eval Type':'Scenario', 'subject':'Subjects'}, inplace=True)
    print(grouped_df)



