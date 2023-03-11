import abc
import logging

import mne
import numpy as np
import pandas as pd
#print("avinash")
from deeb.paradigms.base import BaseParadigm
from deeb.paradigms.erp import N400
from deeb.paradigms.erp import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022

from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
#from .features import AR, PSD

dset = Won2022()
dset.subject_list = dset.subject_list[0:5]
paradigm=P300()
print(paradigm.used_events(dset))
X, features, meta=paradigm.get_data(dset, ar_order=2)
print("features", features)