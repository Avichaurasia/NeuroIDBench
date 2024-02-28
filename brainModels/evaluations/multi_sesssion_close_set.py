import logging
import os
from copy import deepcopy
from time import time
from typing import Optional, Union
import numpy as np
from mne.epochs import BaseEpochs
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GroupKFold,
    cross_val_score,
)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base import BaseEvaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
from scipy.interpolate import interp1d
from ..analysis.metrics import Scores as score
from collections import OrderedDict
from sklearn.utils import shuffle
import mne
import tensorflow as tf
import pickle
import importlib
from .similarity import CalculateSimilarity

log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

#########################################################################################################################################################
##########################################################################################################################################################
                                                    #Close-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################

class MultiSessionCloseSet(BaseEvaluation):

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        return_close_set=True,
        **kwargs
    ):
        self.n_perms = n_perms
        self.data_size = data_size
        self.return_close_set = return_close_set
        super().__init__(**kwargs)

    
##########################################################################################################################################################
##########################################################################################################################################################
                                        #Multiple Session Evaluatiom for State-of-the-algorithms(Close-set Scenario)
##########################################################################################################################################################
##########################################################################################################################################################
    

    def _authenticate_single_subject_close_set(self, X,labels, subject_ids,  pipeline, session_groups=None):

        """
        Perform authentication for a single subject in a multi-session evaluation (close-set scenario) for tradtional 
        shallow classifiers using one vs all strategy

        Parameters:
            X (numpy.ndarray): The input features for authentication.
            y (numpy.ndarray): The labels for the input features.
            pipeline (list): The authentication pipeline including classifier.

        Returns:
            dict: Tuple containing the average authentication scores across the k-fold
                  for each subject.

        This method evaluates the authentication performance for a single subject in a close-set scenario.
        It uses a one vs all strategy to authenticate the subject using the provided features and pipeline.
        Earlier sesions are used for enrollment and later sessions are used for testing.
        EEG data of whole session is used for training and testing.
        """
        accuracy_list=[]
        auc_list=[]
        eer_list=[]
        fpr_list=[]
        tpr_list=[]
        fnr_list=[] 
        frr_1_far_list=[]
        frr_01_far_list=[]
        frr_001_far_list=[]
        classifer=pipeline[-1]
        mean_fpr=np.linspace(0, 1, 100000)

        for enroll_sessions in range(0, len(np.unique(session_groups))-1):

            # Get the session number of the session to be enrolled
            enroll_session=np.unique(session_groups)[enroll_sessions]

            #print("enroll session", enroll_session)

            # Get the indices of the session to be enrolled
            enroll_indices=np.where(session_groups==enroll_session)[0]

            X_train=X[enroll_indices]
            y_train=labels[enroll_indices]

            # Normalizing training and testing data using StandardScaler
            sc=StandardScaler()
            X_train=sc.fit_transform(X_train)
            clf=clone(classifer)

            # Training the model
            model=clf.fit(X_train,y_train)

            #  Iterate over all the sessions except the session to be enrolled
            for test_sessions in range(enroll_sessions+1, len(np.unique(session_groups))):

                # Get the session number of the session to be tested
                test_session=np.unique(session_groups)[test_sessions]

                #print("test session", test_session)

                # Get the indices of the session to be tested
                test_indices=np.where(session_groups==test_session)[0]

                X_test=X[test_indices]
                y_test=labels[test_indices]
                X_test=sc.transform(X_test)

                # Predicting the test set result
                y_pred=model.predict(X_test)
                y_pred_proba=model.predict_proba(X_test)[:,-1]

                eer, frr_1_far, frr_01_far, frr_001_far , auc, inter_tpr=score._calculate_scores(y_pred_proba,y_test, mean_fpr)
                auc_list.append(auc)
                eer_list.append(eer)
                tpr_list.append(inter_tpr)
                frr_1_far_list.append(frr_1_far)
                frr_01_far_list.append(frr_01_far)
                frr_001_far_list.append(frr_001_far)

        average_scores=score._calculate_average_scores(eer_list, frr_1_far_list, frr_01_far_list, frr_001_far_list, auc_list, tpr_list, mean_fpr)
        return average_scores


    def _prepare_data(self, dataset, features, subject_dict):
        """Prepares and combines data from various features for the given dataset.

        Parameters:
            dataset (str): dataset instance.
            features (list): A list of feature objects.

        Returns:
            pandas.DataFrame: A DataFrame containing the combined data from different features.

        Description:
            This function fetches data from each feature in the provided list for a given dataset.
            It concatenates the retrieved data into a single DataFrame. Then, it performs data cleaning
            by removing duplicate columns and filtering out rows with invalid subject and session
            combinations based on a minimum row count criterion.

        """
        df_final=pd.DataFrame()
        for feat in range(0, len(features)-1):
            df=features[feat].get_data(dataset, subject_dict)
            df_final = pd.concat([df_final, df], axis=1)

        if df_final.columns.duplicated().any():
            df_final = df_final.loc[:, ~df_final.columns.duplicated(keep='first')]

        subject_session_counts = df_final.groupby(['subject', 'session']).size().reset_index(name='counts')

        # Identify subjects with sessions having fewer than 4 rows
        invalid_subject_sessions = subject_session_counts[subject_session_counts['counts'] < 4][['subject', 'session']]
        
        # Filter out rows with invalid subject and session combinations
        df_final = df_final[~df_final.set_index(['subject', 'session']).index.isin(invalid_subject_sessions.set_index(['subject', 'session']).index)]
        #print(df_final[['session', 'Subject']].value_counts())

        #print(df[['session', 'Subject']].value_counts())

        return df_final
    
    def traditional_authentication_methods(self, dataset, subject_dict, key, features): 
        """
        Perform traditional authentication methods for single-session close-set evaluation.

        Parameters:
            dataset (Dataset): The dataset to be evaluated.
            subject_dict (dict): A dictionary containing subject information.
            key (str): The key identifier for the authentication method.
            features (list): A list of features used for authentication.

        Returns:
            list: A list of dictionaries containing evaluation metrics for each subject's session.


        Description:
            This method executes traditional authentication methods for single-session close-set evaluation.
            It prepares the data for evaluation and iterates through each subject in the dataset. For each 
            subject, it assigns label 1 to the sessions belonging to the subject being authenticated and 
            label 0 to the rest. The method then authenticates each subject using the provided features and gathers 
            evaluation metrics. Metrics include accuracy, AUC, EER, TPR, among others.
        """ 
        results_close_set=[]
        data=self._prepare_data(dataset, features, subject_dict)
        for subject in tqdm(np.unique(data.subject), desc=f"{key}-MultiSessionCloseSet"):
            df_subj=data.copy(deep=True)
            if not self._valid_sessions(df_subj, subject, dataset):
                continue

            # Assign label 0 to all subjects
            df_subj['Label']=0

            # Updating the label to 1 for the subject being authenticated
            df_subj.loc[df_subj['subject'] == subject, 'Label'] = 1
            session_groups = df_subj.session.values
            subject_ids=df_subj.subject.values
            labels=np.array(df_subj['Label'])
            X=np.array(df_subj.drop(['Label','Event_id','subject','session'],axis=1))

            close_set_scores=self._authenticate_single_subject_close_set(X,labels, subject_ids, features, session_groups=session_groups)
            #mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=close_set_scores

            mean_eer, mean_frr_1_far, mean_frr_01_far, mean_frr_001_far, mean_tpr, mean_auc=close_set_scores
            res_close_set = {
            # "time": duration / 5.0,  # 5 fold CV
            'evaluation': 'Multi Session',
            "eval Type": "Close Set",
            "dataset": dataset.code,
            "pipeline": key,
            "subject": subject,
            "frr_1_far": mean_frr_1_far,
            "frr_0.1_far": mean_frr_01_far,
            "frr_0.01_far": mean_frr_001_far,
           "auc": mean_auc,
            "eer": mean_eer,
           "tpr": mean_tpr,
            "n_samples": len(df_subj)
                }
            #print(res_close_set)
            results_close_set.append(res_close_set)
        return results_close_set

    def _evaluate(self, dataset, pipelines):

        """
        Evaluates authentication methods on a given dataset using specified pipelines.

        Parameters:
            dataset (str): The dataset for evaluation.
            pipelines (dict): A dictionary containing authentication and feature methods as keys
                              and their corresponding features as values.
                              For example: {'AR+PSD+LR': [AutoRegressive(order=6), PowerSpectralDensity(), LogisticRegression()],
                                            'AR+PSD+SVM': [AutoRegressive(order=6), PowerSpectralDensity(), SVC(kernel='rbf', probability=True)],
                                             'Siamese': Siamese()}


        Returns:
            list: A list containing the evaluation results for each specified authentication method.
        """

        X, subject_dict, metadata=self.paradigm.get_data(dataset)
        results_pipeline=[]
        for key, features in pipelines.items():   
            if (key.upper()=='TNN'):
                
                #print("Avinash")
                # If the key is Siamese, then we use the deep learning method
                results=self.deep_learning_method(X, dataset, metadata, key, features)
                results_pipeline.append(results) 
            else:

                # If the key is not Siamese, then we use the traditional authentication methods
                results=self.traditional_authentication_methods(dataset, subject_dict, key, features)
                results_pipeline.append(results)
        return results_pipeline
    
    def evaluate(self, dataset, pipelines):

        """
        Evaluate a dataset using a set of pipelines.

        Parameters:
            - dataset: The dataset for evaluation.
            - pipelines (dict): A dictionary containing authentication and feature methods as keys
                              and their corresponding features as values.
                              For example: {'AR+PSD+LR': [AutoRegressive(order=6), PowerSpectralDensity(), LogisticRegression()],
                                            'AR+PSD+SVM': [AutoRegressive(order=6), PowerSpectralDensity(), SVC()],
                                             'Siamese': Siamese()}
        
        Returns:
            - results: Evaluation results.
            - results_path: Path to save the results.
            - scenario: Evaluation scenario (close-set).
        """

        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for multi session evaluation")
        
        results=self._evaluate(dataset, pipelines)
        scenario="close_Set"
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "MultiSessionEvaluation"
        )
        return results, results_path, scenario
    
    def _valid_subject_samples(self, metadata):

        """
        Filter out subjects with sessions having fewer than 4 samples.

        Parameters:
            - metadata: DataFrame containing metadata.

        Returns:
            - metadata: Filtered metadata.
        """

        subject_session_counts = metadata.groupby(['subject', 'session']).size().reset_index(name='counts')

        # Identify subjects with sessions having fewer than 4 samples
        invalid_subject_sessions = subject_session_counts[subject_session_counts['counts'] < 4][['subject', 'session']]
        
        # Filter out rows with invalid subject and session combinations
        metadata = metadata[~metadata.set_index(['subject', 'session']).index.isin(invalid_subject_sessions.set_index(['subject', 'session']).index)]  
        return metadata
         
    def _valid_sessions(self, df, subject, dataset):

        """
        This function checks if each subject has the same number of sessions.

        Parameters:
        - df: DataFrame containing metadata.

        Returns:
        - True: If each subject has the same number of sessions.
        """
        df_subject=df[df['subject']==subject]
        if (len(df_subject['session'].unique())!=dataset.n_sessions):
            return False
        else:
            return True

    
    def is_valid(self, dataset):
        """
        This function checks if the dataset is appropriate for multi session evaluation.

        Parameters:
        - dataset: The dataset for evaluation.

        Returns:
        - True: If the dataset is appropriate for multi session evaluation.
        """
        return dataset.n_sessions > 1


    