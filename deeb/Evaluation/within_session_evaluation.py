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
    GroupKFold,
    cross_val_score,
)
import pandas as pd
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from deeb.Evaluation.base import BaseEvaluation
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
from scipy.interpolate import interp1d
from deeb.Evaluation.scores import Scores as score
from collections import OrderedDict
from sklearn.utils import shuffle

log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

class WithinSessionEvaluation(BaseEvaluation):
    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        return_close_set=True,
        return_open_set=True,
        **kwargs
    ):
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

    def _authenticate_single_subject_close_set(self, X, y, pipeline):
        accuracy_list=[]
        auc_list=[]
        eer_list=[]
        eer_threshold_list=[]
        fpr_list=[]
        tpr_list=[]
        thresholds_list=[]
        fnr_list=[] 
        frr_1_far_list=[]
        skfold = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=42)
        classifer=pipeline[-1]
        mean_fpr = np.linspace(0, 1, 100)

        # Splitting the dataset into the Training set and Test set
        for fold, (train_index, test_index) in enumerate(skfold.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            X_test=sc.transform(X_test)
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

    def _close_set(self, df_session, pipeline):
        labels=np.array(df_session['Label'])
        X=np.array(df_session.drop(['Label','Event_id','Subject','session'],axis=1))
        return self._authenticate_single_subject_close_set(X,labels, pipeline)


##########################################################################################################################################################
##########################################################################################################################################################
                                                    #Open-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################

   
    def _authenticate_single_subject_open_set(self, imposters_data, imposters_labels, imposter_subject_ids, df_authenticated, pipeline):
        accuracy_list=[]
        auc_list=[]
        eer_list=[]
        eer_threshold_list=[]
        fpr_list=[]
        tpr_list=[]
        thresholds_list=[]
        fnr_list=[] 
        frr_1_far_list=[]
        mean_fpr = np.linspace(0, 1, 100)
        classifier=pipeline[-1]
        groupfold = GroupKFold(n_splits=4)
        for train_index, test_index in groupfold.split(imposters_data, imposters_labels, groups=imposter_subject_ids):
            X_train, X_test = imposters_data[train_index], imposters_data[test_index]
            y_train, y_test = imposters_labels[train_index], imposters_labels[test_index]
            imposter_train, imposter_test=imposter_subject_ids[train_index], imposter_subject_ids[test_index]

            # Assigning 75% samples of authenticated subject to training set
            num_rows = int(len(df_authenticated) * 0.75)
            df_authenticated_train=df_authenticated.sample(n=num_rows)

            # Assigning the remaining 25% samples of authenticated subject to testing set
            df_authenticated_test=df_authenticated.drop(df_authenticated_train.index)

            authenticated_train_lables=np.array(df_authenticated_train['Label'])
            authenticated_train_data=np.array(df_authenticated_train.drop(['Label','Event_id','Subject','session'],axis=1))

            authenticated_test_lables=np.array(df_authenticated_test['Label'])
            authenticated_test_data=np.array(df_authenticated_test.drop(['Label','Event_id','Subject','session'],axis=1))

            X_train = np.concatenate((X_train, authenticated_train_data))
            y_train = np.concatenate((y_train, authenticated_train_lables))
            X_test = np.concatenate((X_test, authenticated_test_data))
            y_test = np.concatenate((y_test, authenticated_test_lables))

            # Shuffle the training and testing data
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
            X_test, y_test = shuffle(X_test, y_test, random_state=42)

            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            X_test=sc.transform(X_test)
            clf=clone(classifier)
           
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
     
    def _open_set(self, df_session, pipeline, subject):
        df_authenticated=df_session[df_session['Subject']==subject]

        # getting the dataframe for rejected subjects
        df_imposters=df_session.drop(df_authenticated.index)

        # getting the subject IDs of the rejected subjects
        imposter_subject_ids = df_imposters.Subject.values

        imposters_labels=np.array(df_imposters['Label'])
        imposters_X=np.array(df_imposters.drop(['Label','Event_id','Subject','session'],axis=1))
        return self._authenticate_single_subject_open_set(imposters_X, imposters_labels, imposter_subject_ids, df_authenticated, pipeline)
        
##########################################################################################################################################################
##########################################################################################################################################################
                                                    #Within Session Evaluation
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

        subject_session_counts = df_final.groupby(['Subject', 'session']).size().reset_index(name='counts')

        # Identify subjects with sessions having fewer than 4 rows
        invalid_subject_sessions = subject_session_counts[subject_session_counts['counts'] < 4][['Subject', 'session']]
        
        # Filter out rows with invalid subject and session combinations
        df_final = df_final[~df_final.set_index(['Subject', 'session']).index.isin(invalid_subject_sessions.set_index(['Subject', 'session']).index)]
        
        # Show value_counts of subject and session
        print(df_final[['session', 'Subject']].value_counts())
        return df_final
    
    def _evaluate(self, dataset, pipelines, param_grid):
        results_close_set=[]
        results_open_set=[]
        for key, features in pipelines.items():
            data=self._prepare_dataset(dataset, features)
            for subject in tqdm(np.unique(data.Subject), desc=f"{key}-WithinSessionEvaluation"):
                df_subj=data.copy(deep=True)
                df_subj['Label']=0
                df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
                for session in np.unique(df_subj.session):
                    df_session= df_subj[df_subj.session==session]

                    if not self._valid_subject(df_session, subject, session):
                        continue

                    if self.return_close_set == False and self.return_open_set==False:
                        message = "Please choose either close-set or open-set scenario for the evaluation"
                        raise ValueError(message)

                    if self.return_close_set:
                        close_set_scores=self._close_set(df_session, pipelines[key])
                        mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=close_set_scores
                        res_close_set = {
                       # "time": duration / 5.0,  # 5 fold CV
                       'evaluation': 'Within Session',
                        "eval Type": "Close Set",
                        "dataset": dataset.code,
                        "pipeline": key,
                        "subject": subject,
                        "session": session,
                        "frr_1_far": mean_frr_1_far,
                        "accuracy": mean_accuracy,
                        "auc": mean_auc,
                        "eer": mean_eer,
                        "tpr": mean_tpr,
                        "tprs_upper": tprs_upper,
                        "tprs_lower": tprr_lower,
                        "std_auc": std_auc,
                         "n_samples": len(df_subj)
                         }
                        results_close_set.append(res_close_set)
                    if self.return_open_set:
                        open_set_scores=self._open_set(df_session, pipelines[key], subject)   
                        mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=open_set_scores

                        res_open_set = {
                       # "time": duration / 5.0,  # 5 fold CV
                       'evaluation': 'Within Session',
                        "eval Type": "Open Set",
                        "dataset": dataset.code,
                        "pipeline": key,
                        "subject": subject,
                        "session": session,
                        "frr_1_far": mean_frr_1_far,
                        "accuracy": mean_accuracy,
                        "auc": mean_auc,
                        "eer": mean_eer,
                        "tpr": mean_tpr,
                        "tprs_upper": tprs_upper,
                        "tprs_lower": tprr_lower,
                        "std_auc": std_auc,
                         "n_samples": len(df_subj)
                        }
                        results_open_set.append(res_open_set)

        if self.return_close_set ==True and self.return_open_set== False:
            scenario='close_set'
            return results_close_set, scenario

        if self.return_close_set ==False and self.return_open_set== True:
            scenario='open_set'
            return results_open_set, scenario
        
        if self.return_close_set ==True and self.return_open_set== True:
            scenario=['close_set', 'open_set']
            return (results_close_set, results_open_set), scenario
        
    def evaluate(self, dataset, pipelines, param_grid):
        results, scenario=self._evaluate(dataset, pipelines, param_grid)
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "WithinSessionEvaluation"
        )
        return results, results_path, scenario
    
    def _valid_subject(self, df, subject, session):
        """Checks if the subject has the required session needed for performing within Session Evaluation"""
        df_subject=df[df['Subject']==subject]
        sessions=df_subject.session.values
        if (session not in sessions):
            return False
        
        else:
            return True
    
    def is_valid(self, dataset):
        return True



    