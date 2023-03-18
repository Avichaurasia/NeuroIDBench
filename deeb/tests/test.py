
import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')
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
from deeb.pipelines.features import AutoRegressive
from deeb.pipelines.features import PowerSpectralDensity
from deeb.pipelines.siamese import Siamese
from deeb.pipelines.base import Basepipeline
from deeb.datasets import utils
from autoreject import AutoReject, get_rejection_threshold
#import tensorflow as tf
#from .features import AR, PSD

dset = BrainInvaders2015a()
dset.subject_list = dset.subject_list[0:5]
paradigm=P300()
#psd=PowerSpectralDensity()
#ar=AutoRegressive(order=4)
print(dset.subject_list)
snn=Siamese(optimizer='Adam')
#print(paradigm.used_events(dset))
#X, sub, meta=paradigm.get_data(dset, return_epochs=False, return_raws=False)
#print("X", sub)
preds, threshold=snn.get_data(dset, paradigm, return_epochs=True, return_raws=False)
print("Threshold", threshold)