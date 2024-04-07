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

class Results():
    """
    A class for storing and managing results in JSON format.

    This class is designed for storing and organizing results generated during experiments. 
    It provides methods for saving, loading, and processing results as JSON files.

    Attributes:
    - None

    Methods:
    - _add_results(results, results_path, scenario):
      Adds results to a JSON file and returns an averaged DataFrame if applicable.

    - _add_dataframe(results_path, scenario):
      Averages results from JSON files and returns them as a DataFrame.

    Example usage:
    ```
    results_handler = Results()
    average_results = results_handler._add_results(results_data, 'results_folder', 'scenario_name')
    ```

    The Results class is typically used to save and manage the outcome of experiments and provides tools for
    aggregating and summarizing results.

    """

    def _add_results(self, dataset, results, results_path, scenario):
        """
        Add results to a JSON file in the specified results folder.

        Parameters:
        - results: Results data to be stored as a JSON file.
        - results_path: The path to the results folder.
        - scenario: A scenario identifier used to create the appropriate JSON file.

        Returns:
        - average_results: A DataFrame of averaged results if multiple JSON files are used.
        """

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

        #getting average results across subjects
        average_results = self._add_dataframe(fname, results_path, scenario)
        return average_results
    
    def _add_dataframe(self, fname, results_path, scenario):
        """
        Average results from JSON files and return them as a DataFrame.

        Parameters:
        - results_path: The path to the results folder.
        - scenario: A scenario identifier used to find the appropriate JSON files.

        Returns:
        - df_results: A DataFrame containing the averaged results.
        """

        if ("close_set" in scenario) and ("open_set" in scenario):
            with open(os.path.join(results_path, "results_close_set.json"), 'r') as f:
                results_list = json.load(f) 
            df_results_close_set=pd.DataFrame(results_list)

            with open(os.path.join(results_path, "results_open_set.json"), 'r') as f:
                results_list_open = json.load(f) 
            df_results_open_set=pd.DataFrame(results_list_open)
            df_results = pd.concat([df_results_close_set, df_results_open_set], ignore_index=True)
        else:
            with open(os.path.join(results_path, fname), 'r') as f:
                results_list = json.load(f) 

            flattened_data = [item for sublist in results_list for item in sublist]
            df_results_close_set=pd.DataFrame(flattened_data)
            df_results = df_results_close_set
        return df_results


