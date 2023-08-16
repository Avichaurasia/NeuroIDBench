import importlib
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob   
import numpy as np
import scipy.signal as scp
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from deeb.datasets.base import BaseDataset
#from moabb.analysis.results import get_string_rep


log = logging.getLogger(__name__)


def create_pipeline_from_config(config):
    """Create a pipeline from a config file.
    takes a config dict as input and return the corresponding pipeline.
    If the pipeline is a Tensorflow pipeline it convert also the optimizer function and the callbacks.
    Parameters
    ----------
    config : Dict.
        Dict containing the config parameters.
    Returns
    -------
    pipeline : Pipeline
        sklearn Pipeline
    """
####################################################################################################################################
################### Below commented code for multiple yaml files or single dataset in single yaml file  ########################################
####################################################################################################################################


    # components = []
    # #count = 0
    # for component in config:
    #     # load the package
    #     #print(count)
    #     #print("component", component)

    #     mod = __import__(component["from"], fromlist=[component["name"]])
    #     #print("mod", mod)
    #     # create the instance
    #     if "parameters" in component.keys():
    #         params = component["parameters"]
    #         if "optimizer" in component["parameters"].keys():
    #             for optm in component["parameters"]["optimizer"]:
    #                 mod_optm = __import__(name=optm["from"], fromlist=[optm["name"]])
    #                 params_optm = optm["parameters"]
    #                 instance = getattr(mod_optm, optm["name"])(**params_optm)
    #                 component["parameters"]["optimizer"] = instance

    #         if "callbacks" in component["parameters"].keys():
    #             cb = []
    #             for callbacks in component["parameters"]["callbacks"]:
    #                 mod_callbacks = __import__(
    #                     name=callbacks["from"], fromlist=[callbacks["name"]]
    #                 )
    #                 params_callbacks = callbacks["parameters"]
    #                 instance = getattr(mod_callbacks, callbacks["name"])(
    #                     **params_callbacks
    #                 )
    #                 cb.append(instance)
    #             component["parameters"]["callbacks"] = cb

    #         if "order" in component['parameters'].keys():
    #             params['order'] = component['parameters']['order']

    #     else:
    #         params = {}
    #     instance = getattr(mod, component["name"])(**params)
    #     components.append(instance)
    #     #count += 1

    # #print("Final components", components)
    # pipeline = make_pipeline(*components)
    # #print("pipeline", pipeline)
    # #print("====================================")
    
    # return pipeline



####################################################################################################################################
############################ updated version for multiple pipelines in a single config file ########################################
####################################################################################################################################

    
    #pipelines=[]
    pipelines=OrderedDict()
    #count = 0
    for name, pipieline in config.items():
        #print("pipelines list", pipelines)
        components = []
        pipelines[name] = {}
        for component in pipieline:
        # load the package
        #print(count)
            #print("component", component)

            mod = __import__(component["from"], fromlist=[component["name"]])
            #print("mod", mod)
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

                if "order" in component['parameters'].keys():
                    params['order'] = component['parameters']['order']

            else:
                params = {}
            instance = getattr(mod, component["name"])(**params)
            components.append(instance)
            #count += 1

        #print("Final components", components)
        pipeline = make_pipeline(*components)
        #print("pipeline", pipeline)
       # print("====================================================")
        #pipelines.append(pipeline)
        pipelines[name] = pipeline
        #print("pipeline", pipeline)
        
    #print("pipelines", pipelines.keys())
    return pipelines


def parse_pipelines_from_directory(dir_path):
    """
    Takes in the path to a directory with pipeline configuration files and returns a dictionary
    of pipelines.
    Parameters
    ----------
    dir_path: str
        Path to directory containing pipeline config .yml or .py files
    Returns
    -------
    pipeline_configs: dict
        Generated pipeline config dictionaries. Each entry has structure:
        'name': string
        'pipeline': sklearn.BaseEstimator
        'paradigms': list of class names that are compatible with said pipeline
    """
    assert os.path.isdir(
        os.path.abspath(dir_path)
    ), "Given pipeline path {} is not valid".format(dir_path)

    # get list of config files
    yaml_files = glob(os.path.join(dir_path, "*.yml"))

    pipeline_configs = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as _file:
            content = _file.read()

            # load config
            config_dict = yaml.load(content, Loader=yaml.FullLoader)
            #print("Avinash", config_dict)
            ppl = create_pipeline_from_config(config_dict["pipeline"])
            if "param_grid" in config_dict:
                pipeline_configs.append(
                    {
                        #"dataset": config_dict["dataset"],
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                        "param_grid": config_dict["param_grid"],
                    }
                )
            else:
                pipeline_configs.append(
                    {
                       # "dataset": config_dict["dataset"],
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                    }
                )

    # we can do the same for python defined pipeline
    # TODO for python pipelines
    python_files = glob(os.path.join(dir_path, "*.py"))
    #print("python_files", python_files)
    for python_file in python_files:
        # print("avinash")
        # print("python_file", python_file)
        spec = importlib.util.spec_from_file_location("custom", python_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        pipeline_configs.append(foo.PIPELINE)
    return pipeline_configs

def _parse_dataset_from_config(config):
    #print("config in _parse_dataset_from_config", config)
    #datasets = []
    #count = 0
    for component in config:
        # load the package
        #print(count)
        #print("component", component)

        mod = __import__(component["from"], fromlist=[component["name"]])
        #print("mod", mod)
        instance = getattr(mod, component["name"])
        instance = instance()
        #print("instance", instance)
        
        #print("mod", mod)
        # create the instance
        if "parameters" in component.keys():
            params = component["parameters"]
            #print("parameters", params.keys())
            # for key, value in params.items():

            if "subjects" in params.keys():
                #subject=int(params['subjects'])
                subject=params['subjects']
                if (type(subject)==list):
                    instance.subject_list=instance.subject_list[subject[0]-1:subject[1]]
                else:
                    subject=int(subject)
                    instance.subject_list=instance.subject_list[:subject]
                
            #     instance.subjects=instance.subjects[:subject]
                #params['subject'] = config['parameters']['subject']

            if "interval" in params.keys():
                instance.interval=params['interval']
                #params['interval'] = config['parameters']['interval']

            if "rejection_threshold" in params.keys():
                instance.rejection_threshold=params['rejection_threshold']
                #params['rejection_threshold'] = config['parameters']['rejection_threshold']

        # else:
        #     params = {}
        # instance = getattr(mod, component["name"])(**params)
        # datasets.append(instance)

    return instance

def parse_pipelines_for_single_dataset(dir_path):
    """
    Takes in the path to a directory with pipeline configuration files and returns a dictionary
    of pipelines.
    Parameters
    ----------
    dir_path: str
        Path to directory containing pipeline config .yml or .py files
    Returns
    -------
    pipeline_configs: dict
        Generated pipeline config dictionaries. Each entry has structure:
        'name': string
        'pipeline': sklearn.BaseEstimator
        'paradigms': list of class names that are compatible with said pipeline
    """
    assert os.path.isdir(
        os.path.abspath(dir_path)
    ), "Given pipeline path {} is not valid".format(dir_path)

    # get list of config files
     # get list of config files
    yaml_files = glob(os.path.join(dir_path, "*.yml"))

    pipeline_configs = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as _file:
            content = _file.read()

            # load config

            #print("content", content)
            config_dict = yaml.load(content, Loader=yaml.FullLoader)
            #print("Avinash", config_dict.keys())    
            #print("Avinash", config_dict['pipelines'])
            dataset= _parse_dataset_from_config(config_dict["dataset"])
            ppl = create_pipeline_from_config(config_dict["pipelines"])
            #print("modified pipeline structure", ppl)
            #print("=====================================")
            if "param_grid" in config_dict:
                pipeline_configs.append(
                    {
                        "dataset": dataset,
                       # "paradigms": config_dict["paradigms"],
                        "pipelines": ppl,
                        "name": config_dict["name"],
                        "param_grid": config_dict["param_grid"],
                    }
                )
            else:
                pipeline_configs.append(
                    {
                       "dataset": dataset,
                       # "paradigms": config_dict["paradigms"],
                        "pipelines": ppl,
                        "name": config_dict["name"],
                    }
                )

    # we can do the same for python defined pipeline
    # TODO for python pipelines
    python_files = glob(os.path.join(dir_path, "*.py"))

    for python_file in python_files:
        spec = importlib.util.spec_from_file_location("custom", python_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        pipeline_configs.append(foo.PIPELINE)
    return pipeline_configs
   
def generate_paradigms(pipeline_configs, context=None, logger=log):
    """
    Takes in a dictionary of pipelines configurations as returned by
    parse_pipelines_from_directory and returns a dictionary of unique paradigms with all pipeline
    configurations compatible with that paradigm.
    Parameters
    ----------
    pipeline_configs:
        dictionary of pipeline configurations
    context:
        TODO:add description
    logger:
        logger
    Returns
    -------
    paradigms: dict
        Dictionary of dictionaries with the unique paradigms and the configuration of the
        pipelines compatible with the paradigm
    """
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


def get_paradigm_from_config(pipeline_configs, context=None, logger=log):
    """
    Takes in a pipeline configuration and returns the unique paradigm
    Parameters
    ----------
    config: dict
        pipeline configuration
    context:
    """
    # context = context or {}
    # paradigms = OrderedDict()
    # paradigms['dataset']=pipeline_configs[0]['dataset']
    # paradigms['pipeline']={}
    # for config in pipeline_configs:
    #     if "dataset" not in config.keys():
    #         logger.error("{} must have a 'dataset' key.".format(config))
    #         continue

    #     if isinstance(config["pipeline"], BaseEstimator):
    #         pipeline = deepcopy(config["pipeline"])
    #     else:
    #         logger.error(config["pipeline"])
    #         raise (ValueError("pipeline must be a sklearn estimator"))
    #     paradigms['pipeline'][config['name']] = pipeline

    # return paradigms

    context = context or {}
    paradigms = OrderedDict()
    paradigms['dataset']=pipeline_configs[0]['dataset']
    paradigms['pipelines']={}
    multi_pipelines = pipeline_configs[0]['pipelines']
    for name, pipeline in multi_pipelines.items():
        if isinstance(pipeline, BaseEstimator):
            pipeline = deepcopy(pipeline)
        else:
            logger.error(pipeline)
            raise (ValueError("pipeline must be a sklearn estimator"))
        paradigms['pipelines'][name] = pipeline

    # paradigms['pipeline']={}
    # for config in pipeline_configs:
    #     if "dataset" not in config.keys():
    #         logger.error("{} must have a 'dataset' key.".format(config))
    #         continue

    #     if isinstance(config["pipeline"], BaseEstimator):
    #         pipeline = deepcopy(config["pipeline"])
    #     else:
    #         logger.error(config["pipeline"])
    #         raise (ValueError("pipeline must be a sklearn estimator"))
    #     paradigms['pipeline'][config['name']] = pipeline

    #print("paradigms", paradigms)
    return paradigms



def generate_param_grid(pipeline_configs, context=None, logger=log):
    context = context or {}
    param_grid = {}
    for config in pipeline_configs:
        if "paradigms" not in config:
            logger.error("{} must have a 'paradigms' key.".format(config))
            continue

        # iterate over paradigms
        if "param_grid" in config:
            param_grid[config["name"]] = config["param_grid"]


    return param_grid
