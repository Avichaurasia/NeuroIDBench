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


class Results():

    def _add_results(results, results_path):
        """Add results to the hdf5 file."""
        if(not osp.exists(results_path)):
            os.makedirs(results_path)
            results.to_csv(osp.join(results_path, "results.csv"), index=False)
            res=results
        else:
            res=pd.read_csv(osp.join(results_path, "results.csv"))
            #print(res)
        result_df=res[['dataset', 'pipeline', 'subject', 'auc', 'eer']]
        return result_df


# def get_string_rep(obj):
#     if issubclass(type(obj), BaseEstimator):
#         str_repr = repr(obj.get_params())
#     else:
#         str_repr = repr(obj)
#     if "<lambda> at " in str_repr:
#         warnings.warn(
#             "You are probably using a classifier with a lambda function"
#             " as an attribute. Lambda functions can only be identified"
#             " by memory address which MOABB does not consider. To avoid"
#             " issues you can use named functions defined using the def"
#             " keyword instead.",
#             RuntimeWarning,
#             stacklevel=2,
#         )
#     str_no_addresses = re.sub("0x[a-z0-9]*", "0x__", str_repr)
#     return str_no_addresses.replace("\n", "").encode("utf8")


# def get_digest(obj):
#     """Return hash of an object repr.
#     If there are memory addresses, wipes them
#     """
#     return hashlib.md5(get_string_rep(obj)).hexdigest()

# class Results:
#     """Class to hold results from the evaluation.evaluate method.
#     Appropriate test would be to ensure the result of 'evaluate' is
#     consistent and can be accepted by 'results.add'
#     Saves dataframe per pipeline and can query to see if particular
#     subject has already been run """

#     def __init__(
#         self,
#         evaluation_class,
#         paradigm_class,
#         suffix="",
#         overwrite=False,
#         hdf5_path=None,
#         additional_columns=None,):
        
#         self.evaluation_class = evaluation_class
#         self.paradigm_class = paradigm_class
#         self.suffix = suffix
#         self.overwrite = overwrite
#         self.additional_columns = additional_columns
#         self._hdf5_path = hdf5_path
#         self._hdf5_file = None

#     def to_dataframe(self, pipelines=None):
#         df_list = []

#         # get the list of pipeline hash
#         digests = []
#         if pipelines is not None:
#             digests = [get_digest(pipelines[name]) for name in pipelines]

#         with h5py.File(self.filepath, "r") as f:
#             for digest, p_group in f.items():
#                 # skip if not in pipeline list
#                 if (pipelines is not None) & (digest not in digests):
#                     continue

#                 name = p_group.attrs["name"]
#                 for dname, dset in p_group.items():
#                     array = np.array(dset["data"])
#                     ids = np.array(dset["id"])
#                     df = pd.DataFrame(array, columns=dset.attrs["columns"])
#                     df["subject"] = [s.decode() for s in ids[:, 0]]
#                     df["session"] = [s.decode() for s in ids[:, 1]]
#                     #df["channels"] = dset.attrs["channels"]
#                     df["n_sessions"] = dset.attrs["n_sessions"]
#                     df["dataset"] = dname
#                     df["pipeline"] = name
#                     df_list.append(df)
#         return pd.concat(df_list, ignore_index=True)

#     def not_yet_computed(self, pipelines, dataset, subj):
#         """Check if a results has already been computed."""
#         ret = {
#             k: pipelines[k]
#             for k in pipelines.keys()
#             if not self._already_computed(pipelines[k], dataset, subj)
#         }
#         return ret

#     def _already_computed(self, pipeline, dataset, subject, session=None):
#         """Check if we have results for a current combination of pipeline
#         / dataset / subject.
#         """
#         with h5py.File(self.filepath, "r") as f:
#             # get the digest from repr
#             digest = get_digest(pipeline)

#             # check if digest present
#             if digest not in f.keys():
#                 return False
#             else:
#                 pipe_grp = f[digest]
#                 # if present, check for dataset code
#                 if dataset.code not in pipe_grp.keys():
#                     return False
#                 else:
#                     # if dataset, check for subject
#                     dset = pipe_grp[dataset.code]
#                     return str(subject).encode("utf-8") in dset["id"][:, 0]