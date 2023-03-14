import abc
import logging

import mne
import numpy as np
import pandas as pd
from deeb.paradigms.base import BaseParadigm
from deeb.paradigms.erp import N400
from deeb.paradigms.erp import P300
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022
from deeb.pipeline.features import AutoRegressive
from deeb.pipeline.features import PowerSpectralDensity
from deeb.pipeline.base import Basepipeline
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
#from .features import AR, PSD

dset = Won2022()
dset.subject_list = dset.subject_list[0:2]
paradigm=P300()
psd=PowerSpectralDensity()
#ar=AutoRegressive()
print(dset.subject_list)
#print(paradigm.used_events(dset))
#X, sub, meta=paradigm.get_data(dset, return_epochs=False, return_raws=False)
#print("X", sub)
features=psd.get_data(dset, paradigm, return_epochs=True, return_raws=False)
print("features", features)