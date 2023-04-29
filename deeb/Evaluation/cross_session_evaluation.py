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
    

    def _authenticate_single_subject_close_set(self, X, labels, pipeline, session_groups=None):
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
        classifer=pipeline[-1]
        mean_fpr = np.linspace(0, 1, 100)
        for train_index, test_index in cv.split(X, labels, groups=session_groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Print the indices of X_train and X_test
            #print("X_train indices", train_index)
            #print("X_test indices", test_index)
            # print("Train data shape", X_train.shape)
            # print("Test data shape", X_test.shape)
            # print("Authenticated train lables", len(np.where(y_train==1)[0]))
            # print("Imposter test labels", len(np.where(y_train==0)[0]))
            # print("Authenticated test labels", len(np.where(y_test==1)[0]))
            # print("Imposter test labels", len(np.where(y_test==0)[0]))


            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            X_test=sc.transform(X_test)

            # Resampling the training data using RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            clf=clone(classifer)

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


    def _close_set(self, df_subj, pipeline):
        # for subject in tqdm(np.unique(data.subject), desc="CrossSession (close-set)"):
        #     df_subj=data.copy(deep=True)
        #     df_subj['Label']=0
        #     df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
        session_groups = df_subj.session.values
        labels=np.array(df_subj['Label'])
        X=np.array(df_subj.drop(['Label','Event_id','Subject','session'],axis=1))
        return self._authenticate_single_subject_close_set(X,labels, pipeline, session_groups=session_groups)
        #return average_scores

#########################################################################################################################################################
##########################################################################################################################################################
                                                    #open-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################

    def _authenticate_single_subject_open_set(self, X,labels, subject_ids,  pipeline, session_groups=None):
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
        classifer=pipeline[-1]
        mean_fpr = np.linspace(0, 1, 100)
        for train_index, test_index in cv.split(X, subject_ids, groups=session_groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            subj_train, subj_test = subject_ids[train_index], subject_ids[test_index]

            # Get authenticated and rejected subjects from training data
            auth_subjects = np.unique(subj_train[y_train == 1])
            rej_subjects = np.unique(subj_train[y_train == 0])
            np.random.shuffle(rej_subjects)

            # Select 75% of rejected subjects for training and 25% for testing
            n_train_rej = int(np.ceil(0.75 * len(rej_subjects)))
            train_rej_subjects = rej_subjects[:n_train_rej]
            test_rej_subjects = rej_subjects[n_train_rej:]

            # Combine authenticated and selected rejected subjects for training and testing
            train_subjects = np.concatenate((auth_subjects, train_rej_subjects))
            test_subjects = np.concatenate((auth_subjects, test_rej_subjects))

            # Get indices for training and testing data
            train_indices = np.isin(subj_train, train_subjects)
            test_indices = np.isin(subj_test, test_subjects)
            
            # Filter data based on indices
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]

            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            X_test=sc.transform(X_test)

            # Resampling the training data using RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            clf=clone(classifer)
            
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
    
    def _open_set(self, df_subj, pipeline):
            # for subject in tqdm(np.unique(data.subject), desc="CrossSession (close-set)"):
            #     df_subj=data.copy(deep=True)
            #     df_subj['Label']=0
            #     df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
        session_groups = df_subj.session.values
        subject_ids=df_subj.Subject.values
        labels=np.array(df_subj['Label'])
        X=np.array(df_subj.drop(['Label','Event_id','Subject','session'],axis=1))
        return self._authenticate_single_subject_open_set(X,labels, subject_ids, pipeline, session_groups=session_groups)

##########################################################################################################################################################
##########################################################################################################################################################
                                                    #Cross Session Evaluation
##########################################################################################################################################################
##########################################################################################################################################################


    def _prepare_dataset(self, dataset, features):
        df_final=pd.DataFrame()
        for feat in range(0, len(features)-1):
            df=features[feat].get_data(dataset, self.paradigm)
            df_final = pd.concat([df_final, df], axis=1)

        # Check if the dataframe contains duplicate columns
        if df_final.columns.duplicated().any():
            df_final = df_final.loc[:, ~df_final.columns.duplicated(keep='first')]

        # Drop rows where "Subject" value_count is less than 4
        subject_counts = df_final["Subject"].value_counts()
        valid_subjects = subject_counts[subject_counts >= 4].index
        df_final = df_final[df_final["Subject"].isin(valid_subjects)]

        return df_final

    def _evaluate(self, dataset, pipelines, param_grid):
        if not self.is_valid(dataset):
            raise ValueError("Dataset is not appropriate for cross session evaluation")  

        results_close_set=[]
        results_open_set=[]  
        for key, features in pipelines.items():
            #data=features[0].get_data(dataset, self.paradigm)
            data=self._prepare_dataset(dataset, features)
            for subject in tqdm(np.unique(data.Subject), desc=f"{key}-CrossSessionEvaluation"):
                df_subj=data.copy(deep=True)
                df_subj['Label']=0
                df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
                
                # Print the value_counts of subjects and sessions in the dataframe
                #print("value_counts", df_subj[['Subject','session']].value_counts())
                #groups = df_subj.session.values

                if self.return_close_set == False and self.return_open_set==False:
                    message = "Please choose either close-set or open-set scenario for the evaluation"
                    raise ValueError(message)

                if self.return_close_set:
                    
                    close_set_scores=self._close_set(df_subj, pipelines[key])
                    mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=close_set_scores
                    res_close_set = {
                       # "time": duration / 5.0,  # 5 fold CV
                        "eval Type": "Close Set",
                        "dataset": dataset.code,
                        "pipeline": key,
                        "subject": subject,
                        #"session": session,
                        "frr_1_far": mean_frr_1_far,
                        "accuracy": mean_accuracy,
                        "auc": mean_auc,
                        "eer": mean_eer,
                        "tpr": mean_tpr,
                        "tprs_upper": tprs_upper,
                        "tprs_lower": tprr_lower,
                        "std_auc": std_auc,
                        #"n_samples": len(data)  # not training sample
                        #"n_channels": data.columns.size
                        }
                    results_close_set.append(res_close_set)

                elif self.return_open_set:
                    #print("groups", groups)
                    open_set_scores=self._open_set(df_subj, pipelines[key])
                    mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=open_set_scores
                    res_open_set = {
                       # "time": duration / 5.0,  # 5 fold CV
                        "eval Type": "Open Set",
                        "dataset": dataset.code,
                        "pipeline": key,
                        "subject": subject,
                        #"session": session,
                        "frr_1_far": mean_frr_1_far,
                        "accuracy": mean_accuracy,
                        "auc": mean_auc,
                        "eer": mean_eer,
                        "tpr": mean_tpr,
                        "tprs_upper": tprs_upper,
                        "tprs_lower": tprr_lower,
                        "std_auc": std_auc,
                        #"n_samples": len(data)  # not training sample
                        #"n_channels": data.columns.size
                        }
                    results_open_set.append(res_open_set)
            
        #return results_close_set, results_open_set

        if self.return_close_set ==True and self.return_open_set== False:
            scenario='close_set'
            return results_close_set, scenario
            #results_close_set=pd.DataFrame(results_close_set)

        if self.return_close_set ==False and self.return_open_set== True:
            scenario='open_set'
            return results_open_set, scenario
        
        if self.return_close_set ==True and self.return_open_set== True:
            scenario=['close_set', 'open_set']
            return (results_close_set, results_open_set), scenario
    
    def evaluate(self, dataset, pipelines, param_grid):
        #yield from self._evaluate(dataset, pipelines, param_grid)
        results, scenario=self._evaluate(dataset, pipelines, param_grid)
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "CrossSessionEvaluation"
            #f"{dataset.code}_CloseSetEvaluation")
        )
        return results, results_path, scenario

    def is_valid(self, dataset):
        return dataset.n_sessions > 1 