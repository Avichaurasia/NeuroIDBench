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
log = logging.getLogger(__name__)

def create_pipeline_from_config(config):

    """
    Create a pipeline from a config file.
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
    pipelines=OrderedDict()
    for name, pipieline in config.items():
        components = []
        pipelines[name] = {}
        for component in pipieline:

        # load the package
            mod = __import__(component["from"], fromlist=[component["name"]])
            if "parameters" in component.keys():
                params = component["parameters"]

                if "user_tnn_path" in component["parameters"].keys():
                    params["user_tnn_path"] = component["parameters"]["user_tnn_path"]
                    
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
        pipeline = make_pipeline(*components)
        pipelines[name] = pipeline
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
                       # "dataset": config_dict["dataset"],
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                    }
                )
    python_files = glob(os.path.join(dir_path, "*.py"))
    for python_file in python_files:
        spec = importlib.util.spec_from_file_location("custom", python_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        pipeline_configs.append(foo.PIPELINE)
    return pipeline_configs

def _parse_dataset_from_config(config):

    """
    Parse dataset configuration from a given configuration dictionary.

    This function takes a configuration dictionary and parses the dataset component based on the provided
    configuration. It dynamically imports the specified dataset class and initializes it with the given parameters.

    Parameters:
    - config (dict): A configuration dictionary containing information about the dataset component, including
      class name, parameters, and settings.

    Returns:
    - instance: An instance of the specified dataset class with the provided settings.

    Example configuration:
    ```
    config = {
        "from": "my_dataset_module",
        "name": "MyDataset",
        "parameters": {
            "subjects": [1, 10],
            "interval": [0, 2],
            "rejection_threshold": 0.5
        }
    }
    dataset_instance = _parse_dataset_from_config(config)
    ```

    This function allows dynamic loading and configuration of dataset classes based on the provided configuration.
    """

    for component in config:
        mod = __import__(component["from"], fromlist=[component["name"]])
        instance = getattr(mod, component["name"])
        instance = instance()
        
        # create the instance
        if "parameters" in component.keys():
            params = component["parameters"]
            if "subjects" in params.keys():
                subject=params['subjects']
                if (type(subject)==list):
                    instance.subject_list=instance.subject_list[subject[0]-1:subject[1]]
                else:
                    subject=int(subject)
                    instance.subject_list=instance.subject_list[:subject]
            if "interval" in params.keys():
                instance.interval=params['interval']

            if "rejection_threshold" in params.keys():
                instance.rejection_threshold=params['rejection_threshold']
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

    yaml_files = glob(os.path.join(dir_path, "*.yml"))

    pipeline_configs = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as _file:
            content = _file.read()
            config_dict = yaml.load(content, Loader=yaml.FullLoader)
            dataset= _parse_dataset_from_config(config_dict["dataset"])
            ppl = create_pipeline_from_config(config_dict["pipelines"])
            if "param_grid" in config_dict:
                pipeline_configs.append(
                    {
                        "dataset": dataset,
                        "pipelines": ppl,
                        "name": config_dict["name"],
                        "param_grid": config_dict["param_grid"],
                    }
                )
            else:
                pipeline_configs.append(
                    {
                       "dataset": dataset,
                        "pipelines": ppl,
                        "name": config_dict["name"],
                    }
                )
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
