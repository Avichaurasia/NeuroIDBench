import hashlib
import os
import os.path as osp
import re
import warnings
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from sklearn.base import BaseEstimator
import h5py
import json
from numpyencoder import NumpyEncoder
import datetime

class Results():
    """
        class that will abstract result storage
    """

    def _add_results(self, results, results_path, scenario):
        """Add results dataframe to path dataset.datasetpath as json file."""

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if ("close_set" in scenario) and ("open_set" in scenario):
            results_close_set, results_open_set = results
            with open(os.path.join(results_path, "results_close_set.json"), 'w') as f:
                json.dump(results_close_set, f, cls=NumpyEncoder)

            with open(os.path.join(results_path, "results_open_set.json"), 'w') as f:
                json.dump(results_open_set, f, cls=NumpyEncoder)
        else:
            fname="results_"+scenario+".json"
            with open(os.path.join(results_path, fname), 'w') as f:
                json.dump(results, f, cls=NumpyEncoder)

        # getting average results across subjects
        average_results = self._add_dataframe(results_path, scenario)
        return average_results
    
    def _add_dataframe(self, results_path, scenario):
        """Average results across subjects."""

        if ("close_set" in scenario) and ("open_set" in scenario):
            with open(os.path.join(results_path, "results_close_set.json"), 'r') as f:
                results_list = json.load(f) 
            df_results_close_set=pd.DataFrame(results_list)

            with open(os.path.join(results_path, "results_open_set.json"), 'r') as f:
                results_list_open = json.load(f) 
            df_results_open_set=pd.DataFrame(results_list_open)
            df_results = pd.concat([df_results_close_set, df_results_open_set], ignore_index=True)
        else:
            fname="results_"+scenario+".json"
            with open(os.path.join(results_path, fname), 'r') as f:
                results_list = json.load(f) 
            df_results_close_set=pd.DataFrame(results_list)
            df_results = df_results_close_set
        return df_results


