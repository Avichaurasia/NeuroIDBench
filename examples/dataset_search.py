import sys
#sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


from deeb.datasets import brainInvaders15a, mantegna2019, utils
#from deeb.evaluation.evaluation import CloseSetEvaluation
from deeb.paradigms.p300 import P300
from deeb.paradigms.p300 import N400
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
#from deeb.evaluation import (CloseSetEvaluation, OpenSetEvaluation,)
log = logging.getLogger(__name__)

#print(os.listdir("./pipelines"))
# with open ("../pipelines/AR_SVM.yml", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         print(line, end="")


def create_pipeline_from_config(config):
    """Create a pipeline from a config file."""

    components = []
    #print("config: ", config)
    for component in config:
        # load the package
        mod = __import__(component["from"], fromlist=[component["name"]])
        # create the instance
        if "parameters" in component.keys():
            params = component["parameters"]
            if "optimizer" in component["parameters"].keys():
                for optm in component["parameters"]["optimizer"]:
                    mod_optm = __import__(name=optm["from"], fromlist=[optm["name"]])
                    params_optm = optm["parameters"]
                    instance = getattr(mod_optm, optm["name"])(**params_optm)
                    component["parameters"]["optimizer"] = instance

            if "callbacks" in component["parameters"].keys():
                cb = []
                for callbacks in component["parameters"]["callbacks"]:
                    mod_callbacks = __import__(
                        name=callbacks["from"], fromlist=[callbacks["name"]]
                    )
                    params_callbacks = callbacks["parameters"]
                    instance = getattr(mod_callbacks, callbacks["name"])(
                        **params_callbacks
                    )
                    cb.append(instance)
                component["parameters"]["callbacks"] = cb

        else:
            params = {}
        instance = getattr(mod, component["name"])(**params)
        components.append(instance)

    pipeline = make_pipeline(*components)
    return pipeline

def generate_paradigms(pipeline_configs, context=None, logger=log):
    context = context or {}
    paradigms = OrderedDict()
    for config in pipeline_configs:
        if "paradigms" not in config.keys():
            logger.error("{} must have a 'paradigms' key.".format(config))
            continue

        # iterate over paradigms

        for paradigm in config["paradigms"]:
            # check if it is in the context parameters file
            if len(context) > 0:
                if paradigm not in context.keys():
                    logger.debug(context)
                    logger.warning(
                        "Paradigm {} not in context file {}".format(
                            paradigm, context.keys()
                        )
                    )

            if isinstance(config["pipeline"], BaseEstimator):
                pipeline = deepcopy(config["pipeline"])
            else:
                logger.error(config["pipeline"])
                raise (ValueError("pipeline must be a sklearn estimator"))

            # append the pipeline in the paradigm list
            if paradigm not in paradigms.keys():
                paradigms[paradigm] = {}

            # FIXME name are not unique
            #logger.debug("Pipeline: \n\n {} \n".format(get_string_rep(pipeline)))
            paradigms[paradigm][config["name"]] = pipeline

    return paradigms

def parse_pipelines_from_directory(dir_path):
    assert os.path.isdir(
        os.path.abspath(dir_path)
    ), "Given pipeline path {} is not valid".format(dir_path)

    # get list of config files
    yaml_files = glob(os.path.join(dir_path, "*.yml"))

    pipeline_configs = []
    for yaml_file in yaml_files:

        # Reading the yaml file
        with open(yaml_file, "r") as _file:
            content = _file.read()
            #print("content", content)

            # load config
            config_dict = yaml.load(content, Loader=yaml.FullLoader)

            # Create dictionary of pipeline from .yaml files
            ppl = create_pipeline_from_config(config_dict["pipeline"])
            if "param_grid" in config_dict:
                pipeline_configs.append(
                    {
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                        "param_grid": config_dict["param_grid"],
                    }
                )
            else:
                pipeline_configs.append(
                    {
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                    }
                )

    # we can do the same for python defined pipeline
    # TODO for python pipelines
    #print("dir_path", dir_path)
    python_files = glob(os.path.join(dir_path, "*.py"))
    #print("python_files", python_files)
    for python_file in python_files:
        #print("python_file", python_file)
        spec = importlib.util.spec_from_file_location("custom", python_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        pipeline_configs.append(foo.PIPELINE)
    
    #print("pipeline_configs", pipeline_configs)
    return pipeline_configs


def benchmark(pipelines="../pipelines",
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
            data = p.datasets

            # Selecting 10 subjects from each dataset and then return the updated datasets object
            datasets = []
            for i in data:
                i.subject_list = i.subject_list[:10]
                datasets.append(i)
            
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
    result=benchmark()
    print(result)

