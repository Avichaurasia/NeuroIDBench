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
#from scipy.optimize import brentq
from scipy.interpolate import interp1d
from deeb.Evaluation.scores import Scores as score
from collections import OrderedDict

log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

class WithinSessionEvaluation(BaseEvaluation):
    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        # dataset=None,
        return_close_set=True,
        return_open_set=True,
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

        # Defining the Stratified KFold
        skfold = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=42)
        classifer=pipeline[-1]
        #classifier=pipeline.steps[1:]
        #print("classifer",classifer)
        mean_fpr = np.linspace(0, 1, 100)

        # Splitting the dataset into the Training set and Test set
        for fold, (train_index, test_index) in enumerate(skfold.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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
    

    def _close_set(self, df_session, pipeline):
        # for session in np.unique(df_subj.session):
        #     df_session= df_subj[df_subj.session==session]
        labels=np.array(df_session['Label'])
        X=np.array(df_session.drop(['Label','Event_id','Subject','session'],axis=1))
        #print("X",X.shape)
        return self._authenticate_single_subject_close_set(X,labels, pipeline)
        #return average_scores


##########################################################################################################################################################
##########################################################################################################################################################
                                                    #Open-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################


    def _build_model(self, train_set, test_set, classifier, mean_fpr, param_grid=None):
        X_train=train_set.drop(['Subject', 'Event_id',"session",'Label'], axis=1)
        X_train=np.array(X_train)
        y_train=np.array(train_set['Label'])

        # Diving training data into X_test and y_test
        X_test=test_set.drop(['Subject', 'Event_id',"session",'Label'], axis=1)
        X_test=np.array(X_test)
        y_test=np.array(test_set['Label'])

         # Normalizing the data using StandardScaler
        sc=StandardScaler()
        X_train=sc.fit_transform(X_train)
        X_test=sc.transform(X_test)

        # Resampling the data using RandomOverSampler
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        #model=pipeline.fit(X_train, y_train)

        clf=clone(classifier)
        # Training the model
        model=clf.fit(X_train,y_train)

        # Predicting the test set result
        y_pred=model.predict(X_test)
        y_pred_proba=model.predict_proba(X_test)[:,-1]
        accuracy=accuracy_score(y_test,y_pred)
        auc, eer, eer_theshold, inter_tpr, tpr, fnr, frr_1_far=score._calculate_scores(y_pred_proba,y_test, mean_fpr)
        return (accuracy, auc, eer, eer_theshold, inter_tpr, tpr, fnr, frr_1_far)

    def _authenticate_single_subject_open_set(self, df, df_authenticated, df_rejected, subject_ids, pipeline, k):
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
        for fold in range(k):

            #Randomly selecing 75% subjects from the rejected subjects
            train_subject_ids = random.sample(subject_ids, k=int(len(subject_ids) * 0.75))

            #Selecting the remaining subjects as test subjects which are not part of training data
            test_subject_ids=df[~df['Subject'].isin(train_subject_ids)]['Subject'].unique()
            test_subject_ids=list(test_subject_ids)
            
            # Divide the dataframe of rejected subjects into training and testing sets based on subject ids
            train_set = df_rejected[df_rejected['Subject'].isin(train_subject_ids)]
            test_set = df_rejected[df_rejected['Subject'].isin(test_subject_ids)]
            
            # Adding Authenticated subjects data in the training as well testing

            # Assigning 75% samples of authenticated subject to training set
            num_rows = int(len(df_authenticated) * 0.75)
            df_authenticated_train=df_authenticated.sample(n=num_rows)

            # Assigning the remaining 25% samples of authenticated subject to testing set
            df_authenticated_test=df_authenticated.drop(df_authenticated_train.index)
            
            train_set=pd.concat([df_authenticated_train, train_set], axis=0)
            test_set=pd.concat([df_authenticated_test, test_set], axis=0)
            accuracy, auc, eer, eer_theshold, inter_tpr, tpr, fnr, frr_1_far=self._build_model(train_set, test_set, classifier, mean_fpr)
            accuracy_list.append(accuracy)
            auc_list.append(auc)
            eer_list.append(eer)
            tpr_list.append(inter_tpr)
            fnr_list.append(fnr)
            frr_1_far_list.append(frr_1_far)
        average_scores=score._calculate_average_scores(accuracy_list, tpr_list, eer_list, mean_fpr, auc_list, frr_1_far_list)
        return average_scores

    def _open_set(self, df_session, pipeline, subject):
        # for subject in tqdm(np.unique(df.subject), desc="WithinSession (open-set)"):
        #     df_subj=df.copy(deep=True)
        #     df_subj['Label']=0
        #     df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1

        df_authenticated=df_session[df_session['Subject']==subject]

        # getting the dataframe for rejected subjects
        df_rejected=df_session.drop(df_authenticated.index)

        # getting the subject IDs of the rejected subjects
        subject_ids = list(set(df_rejected['Subject']))
        return self._authenticate_single_subject_open_set(df_session, df_authenticated, df_rejected, subject_ids, pipeline, k=6)

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

        # Drop rows where "Subject" value_count is less than 4
        subject_counts = df_final["Subject"].value_counts()
        valid_subjects = subject_counts[subject_counts >= 4].index
        df_final = df_final[df_final["Subject"].isin(valid_subjects)]

        return df_final
    
    def _evaluate(self, dataset, pipelines, param_grid):
        results_close_set=[]
        results_open_set=[]
        #print("len of pipelines", pipelines.keys())
        for key, features in pipelines.items():
            data=self._prepare_dataset(dataset, features)
            #print("data", data)
            #data=features[0].get_data(dataset, self.paradigm)
            for subject in tqdm(np.unique(data.Subject), desc=f"{key}-WithinSessionEvaluation"):
                df_subj=data.copy(deep=True)
                df_subj['Label']=0
                df_subj.loc[df_subj['Subject'] == subject, 'Label'] = 1
                for session in np.unique(df_subj.session):
                    df_session= df_subj[df_subj.session==session]

                    if self.return_close_set == False and self.return_open_set==False:
                        message = "Please choose either close-set or open-set scenario for the evaluation"
                        raise ValueError(message)

                    if self.return_close_set:
                        close_set_scores=self._close_set(df_session, pipelines[key])
                        mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=close_set_scores
                        res_close_set = {
                       # "time": duration / 5.0,  # 5 fold CV
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
                        #"n_samples": len(data)  # not training sample
                        #"n_channels": data.columns.size
                         }
                        results_close_set.append(res_close_set)
                    
                        # print("I am done with close set")
                        # print("return open set", self.return_open_set)
                    if self.return_open_set:
                        #print("open set")
                        open_set_scores=self._open_set(df_session, pipelines[key], subject)   
                        mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=open_set_scores

                        res_open_set = {
                       # "time": duration / 5.0,  # 5 fold CV
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
                        #"n_samples": len(data)  # not training sample
                        #"n_channels": data.columns.size
                        }
                        results_open_set.append(res_open_set)

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
            #results_open_set=pd.DataFrame(results_open_set)

            # results_close_set=pd.DataFrame(results_close_set)
            # results_close_set=results_close_set.sort_values(by=['dataset', 'pipeline', 'subject', 'session'])
                       
        #return results_close_set, results_open_set
    
    def evaluate(self, dataset, pipelines, param_grid):
        #yield from self._evaluate(dataset, pipelines, param_grid)
        results, scenario=self._evaluate(dataset, pipelines, param_grid)
        #print(type(results))
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "WithinSessionEvaluation"
            #f"{dataset.code}_CloseSetEvaluation")
        )
        return results, results_path, scenario
    
    def is_valid(self, dataset):
        return True



    