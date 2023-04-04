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
    cross_val_score,
)
import pandas as pd
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from deeb.evaluation.base import BaseEvaluation
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from deeb.evaluation.scores import Scores as score
from collections import OrderedDict

log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]


class CloseSetEvaluation(BaseEvaluation):
    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        # dataset=None,
        # paradigm=None,
        #paradigm=None,
        **kwargs
    ):
        # self.dataset = dataset
        # self.paradigm = paradigm
        #self.paradigm = paradigm
        self.n_perms = n_perms
        self.data_size = data_size
        super().__init__(**kwargs)
        
    def _authenticate_single_subject(self, X,y, pipeline, param_grid=None):
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
        skfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
        classifer=pipeline[1:]
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
            oversampler = RandomOverSampler()
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
        mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=score._calculate_average_scores(
                                                                                    accuracy_list, tpr_list, eer_list, mean_fpr, auc_list, frr_1_far_list)
        return (mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far)
         
    def _evaluate(self, dataset, pipelines, param_grid):
        results=[]
        #dict_results_list = []
        
        for key, features in pipelines.items():
            #dict_results={}
            #print("Items inside idctionary", features)
            data=features[0].get_data(dataset, self.paradigm)
            for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CloseSetEvaulation"):
                df=data.copy(deep=True)
                #session=dataset.get_session(subject)
                df['Label']=0
                df.loc[df['Subject'] == subject, 'Label'] = 1
                labels=np.array(df['Label'])
                X=np.array(df.drop(['Label','Event_id','Subject','Session'],axis=1))
                mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=self._authenticate_single_subject(
                                                                                                                X,labels, pipelines[key], param_grid)
                
                res = {
                       # "time": duration / 5.0,  # 5 fold CV
                        "dataset": dataset.code,
                        "pipeline": key,
                        "subject": subject,
                       "session": 1,
                       "frr_1_far": mean_frr_1_far,
                       "accuracy": mean_accuracy,
                        "auc": mean_auc,
                        "eer": mean_eer,
                        "tpr": mean_tpr,
                        "tprs_upper": tprs_upper,
                        "tprs_lower": tprr_lower,
                        "std_auc": std_auc,
                        "n_samples": len(data)  # not training sample
                        #"n_channels": data.columns.size
                        
                    }
                
                # print environment variables
                #print(f"Environment variables: {os.environ}") 

                # dict_results=OrderedDict()
                # dict_results["dataset"]=dataset.code
                # dict_results["pipeline"]=key
                # dict_results["subject"]=subject
                # dict_results["session"]=1
                # dict_results["accuracy"]=mean_accuracy
                # dict_results["auc"]=mean_auc
                # dict_results["eer"]=mean_eer
                # dict_results["tpr"]=mean_tpr
                # dict_results["tprs_upper"]=tprs_upper
                # dict_results["tprs_lower"]=tprr_lower
                # dict_results["std_auc"]=std_auc
                # dict_results["n_samples"]=len(data)
                # #dict_results["n_channels"]=data.columns.size

                #dict_results_list.append(dict_results)
                results.append(res)
        return results

    def evaluate(self, dataset, pipelines, param_grid):
        #yield from self._evaluate(dataset, pipelines, param_grid)
        results=self._evaluate(dataset, pipelines, param_grid)
        #print(type(results))
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "CloseSetEvaluation"
            #f"{dataset.code}_CloseSetEvaluation")
        )


        #print(results)
        return results, results_path
        #return pd.DataFrame(results), results_path, dict_results_list

    def is_valid(self, dataset):
        return True
    
   
class OpenSetEvaluation(BaseEvaluation):
    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        **kwargs,
    ):
        self.data_size = data_size
        self.n_perms = n_perms
        self.calculate_learning_curve = self.data_size is not None
        super().__init__(**kwargs)

    def _build_model(self, train_set, test_set, classifier, mean_fpr, param_grid=None):
        X_train=train_set.drop(['Subject', 'Event_id',"Session",'Label'], axis=1)
        X_train=np.array(X_train)
        y_train=np.array(train_set['Label'])

        # Diving training data into X_test and y_test
        X_test=test_set.drop(['Subject', 'Event_id',"Session",'Label'], axis=1)
        X_test=np.array(X_test)
        y_test=np.array(test_set['Label'])

         # Normalizing the data using StandardScaler
        sc=StandardScaler()
        X_train=sc.fit_transform(X_train)
        X_test=sc.transform(X_test)

        # Resampling the data using RandomOverSampler
        oversampler = RandomOverSampler()
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

    def _authenticate_single_subject(self, df, df_authenticated, df_rejected, subject_ids, pipeline, param_grid=None, k=4):
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
        classifier=pipeline[1:]
        for fold in range(k):

            #Randomly selecing 75% subjects from the rejected subjects
            train_subject_ids = random.sample(subject_ids, k=int(len(subject_ids) * 0.75))

            #Selecting the remaining subjects as test subjects which are not part of training data
            test_subject_ids=df[~df['Subject'].isin(train_subject_ids)]['Subject'].unique()
            test_subject_ids=list(test_subject_ids)
            
            # Divide the dataframe of rejected subjects into training and testing sets based on subject id
            train_set = df_rejected[df_rejected['Subject'].isin(train_subject_ids)]
            test_set = df_rejected[df_rejected['Subject'].isin(test_subject_ids)]
            
            # Adding Authenticated subjects data in the training as well testing

            # Assigning 75% samples of authenticated subject to training set
            num_rows = int(len(df_authenticated) * 0.75)
            # Assigning the remaining 25% samples of authenticated subject to testing set
            df_authenticated_train=df_authenticated.sample(n=num_rows)
            df_authenticated_test=df_authenticated.drop(df_authenticated_train.index)
            
            train_set=pd.concat([df_authenticated_train, train_set], axis=0)
            test_set=pd.concat([df_authenticated_test, test_set], axis=0)
            accuracy, auc, eer, eer_theshold, inter_tpr, tpr, fnr, frr_1_far=self._build_model(train_set, test_set, classifier, mean_fpr, param_grid)
            accuracy_list.append(accuracy)
            auc_list.append(auc)
            eer_list.append(eer)
            tpr_list.append(inter_tpr)
            fnr_list.append(fnr)
            frr_1_far_list.append(frr_1_far)
        mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=score._calculate_average_scores(
                                                                                    accuracy_list, tpr_list, eer_list, mean_fpr, auc_list, frr_1_far_list)
        
        # print("before dictionary", type(tprr_lower))
        # print(type(tprs_upper))
        return (mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far)
         
    def _evaluate(self, dataset, pipelines, param_grid):
        results=[]
        for key, features in pipelines.items():
            data=features[0].get_data(dataset, self.paradigm)
            
            for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-OpenSetEvaulation"):
                df=data.copy(deep=True)
                df['Label']=0
                df.loc[df['Subject'] == subject, 'Label'] = 1

                df_authenticated=df[df['Subject']==subject]

                # getting the dataframe for rejected subjects
                df_rejected=df.drop(df_authenticated.index)

                # getting the subject IDs of the rejected subjects
                subject_ids = list(set(df_rejected['Subject']))
                mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=self._authenticate_single_subject(df, 
                                                                df_authenticated, df_rejected, subject_ids, pipelines[key], param_grid, k=4) 
                
                
                res = {
                        "dataset": dataset.code,
                        "pipeline": key,
                        "subject": subject,
                    "session": 1,
                    "frr_1_far": mean_frr_1_far,
                    "accuracy": mean_accuracy,
                        "auc": mean_auc,
                        "eer": mean_eer,
                        "tpr": mean_tpr,
                        "tprs_upper": tprs_upper,
                        "tprs_lower": tprr_lower,
                        "std_auc": std_auc,
                        "n_samples": len(data)  # not training sample
                        #"n_channels": data.columns.size     
                    }
                # print("mean_auc", mean_auc)
                # print("mean_eer", mean_eer)
                # print("After dictionary", type(res['tprr_lower']))
                # print(type(res['tprr_upper']))


                results.append(res)

            # except Exception as e:
            #     print(f"An error occurred: {e}")
            #     #continue

        #print(pd.DataFrame(results))
        #print(results[0])
        return results

    def evaluate(self, dataset, pipelines, param_grid):
        # if self.calculate_learning_curve:
        #     yield from self._evaluate_learning_curve(dataset, pipelines)
        # else:
        #yield from self._evaluate(dataset, pipelines, param_grid)

        results=self._evaluate(dataset, pipelines, param_grid)
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "OpenSetEvaluation",
            #f"{dataset.code}_OpenSetEvaluation")
        )
        
        return results, results_path

    def is_valid(self, dataset):
        return True
    