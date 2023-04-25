import logging
import os
from copy import deepcopy
from time import time
from typing import Optional, Union

import joblib
import numpy as np
from mne.epochs import BaseEpochs
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    RepeatedStratifiedKFold,
    cross_val_score,
)
import pandas as pd
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from deeb.Evaluation.base import BaseEvaluation
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from deeb.Evaluation.scores import Scores as score
from collections import OrderedDict

log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

class CrossSessionEvaluation(BaseEvaluation):
    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        # dataset=None,
        return_close_set: bool = True,
        return_open_set: bool = True,
        # paradigm=None,
        #paradigm=None,
        **kwargs
    ):
        # self.dataset = dataset
        # self.paradigm = paradigm
        #self.paradigm = paradigm
        self.n_perms = n_perms
        self.data_size = data_size
        self.return_close_set = return_close_set
        self.return_open_set = return_open_set
        super().__init__(**kwargs) 

#########################################################################################################################################################
##########################################################################################################################################################
                                                    #Close-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################
    

    def _authenticate_single_subject_close_set(self, X, labels, pipeline, groups=None):
        accuracy_list=[]
        auc_list=[]
        eer_list=[]
        eer_threshold_list=[]
        fpr_list=[]
        tpr_list=[]
        thresholds_list=[]
        fnr_list=[] 
        frr_1_far_list=[]
        cv = LeaveOneGroupOut()
        classifer=pipeline[1:]
        mean_fpr = np.linspace(0, 1, 100)
        for train_index, test_index in cv.split(X, labels, groups=groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            X_test=sc.transform(X_test)

            # Resampling the training data using RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            clf=clone(classifer)
            #print("cloned classifer", clf)
            # Training the model
            model=clf.fit(X_train,y_train)

            # Predicting the test set result
            y_pred=model.predict(X_test)
            y_pred_proba=model.predict_proba(X_test)[:,-1]

            # calculating auc, eer, eer_threshold, fpr, tpr, thresholds for each k-fold
            auc, eer, eer_theshold, inter_tpr, tpr, fnr, frr_1_far=score._calculate_scores(y_pred_proba,y_test, mean_fpr)
            accuracy_list.append(accuracy_score(y_test,y_pred))
            auc_list.append(auc)
            eer_list.append(eer)
            tpr_list.append(inter_tpr)
            fnr_list.append(fnr)
            frr_1_far_list.append(frr_1_far)
            average_scores=score._calculate_average_scores(accuracy_list, tpr_list, eer_list, mean_fpr, auc_list, frr_1_far_list)
        return average_scores


    def _close_set(self, data, pipeline, groups=None):
        for subject in tqdm(np.unique(data.subject), desc="CrossSession (close-set)"):
            df_subj=data.copy(deep=True)
            df_subj['Label']=0
            df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
            labels=np.array(df_subj['Label'])
            X=np.array(df_subj.drop(['Label','Event_id','Subject','Session'],axis=1))
            average_scores=self._authenticate_single_subject_close_set(X,labels, pipeline, groups=groups)

#########################################################################################################################################################
##########################################################################################################################################################
                                                    #open-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################


    # Need to make the open-set scenario for this where subjects used in the training from one sessions doesn't get used in testing
    def _authenticate_single_subject_open_set(X,labels, pipeline, groups=None):
        accuracy_list=[]
        auc_list=[]
        eer_list=[]
        eer_threshold_list=[]
        fpr_list=[]
        tpr_list=[]
        thresholds_list=[]
        fnr_list=[] 
        frr_1_far_list=[]
        cv = LeaveOneGroupOut()
        classifer=pipeline[1:]
        mean_fpr = np.linspace(0, 1, 100)
        for train_index, test_index in cv.split(X, labels, groups=groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            X_test=sc.transform(X_test)

            # Resampling the training data using RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            clf=clone(classifer)
            #print("cloned classifer", clf)
            # Training the model
            model=clf.fit(X_train,y_train)

            # Predicting the test set result
            y_pred=model.predict(X_test)
            y_pred_proba=model.predict_proba(X_test)[:,-1]

            # calculating auc, eer, eer_threshold, fpr, tpr, thresholds for each k-fold
            auc, eer, eer_theshold, inter_tpr, tpr, fnr, frr_1_far=score._calculate_scores(y_pred_proba,y_test, mean_fpr)
            accuracy_list.append(accuracy_score(y_test,y_pred))
            auc_list.append(auc)
            eer_list.append(eer)
            tpr_list.append(inter_tpr)
            fnr_list.append(fnr)
            frr_1_far_list.append(frr_1_far)
            average_scores=score._calculate_average_scores(accuracy_list, tpr_list, eer_list, mean_fpr, auc_list, frr_1_far_list)
        return average_scores
    
    def _open_set(self, data, pipeline, groups=None):
            for subject in tqdm(np.unique(data.subject), desc="CrossSession (close-set)"):
                df_subj=data.copy(deep=True)
                df_subj['Label']=0
                df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
                labels=np.array(df_subj['Label'])
                X=np.array(df_subj.drop(['Label','Event_id','Subject','Session'],axis=1))
                average_scores=self._authenticate_single_subject_open_set(X,labels, pipeline, groups=groups)

            return average_scores

##########################################################################################################################################################
##########################################################################################################################################################
                                                    #Cross Session Evaluation
##########################################################################################################################################################
##########################################################################################################################################################

    def evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for cross session evaluation")  

        results_close_set=[]
        results_open_set=[]  
        for key, features in pipelines.items():
            data=features[0].get_data(dataset, self.paradigm)
            groups = data.session.values

            if self.return_close_set:
                close_set_scores=self._close_set(data, y, pipelines[key], groups=groups)

            elif self.return_open_set:
                open_set_scores=self._open_set(data, pipelines[key], groups=groups)  

    def is_valid(self, dataset):
        return dataset.n_sessions > 1 