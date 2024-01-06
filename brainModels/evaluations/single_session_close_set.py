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
from .metrics import Scores as score
from collections import OrderedDict
from sklearn.utils import shuffle
import mne
import tensorflow as tf
import pickle
import importlib
from .similarity import CalculateSimilarity
import gc

log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

#########################################################################################################################################################
##########################################################################################################################################################
                                                    #Close-set Scenario
##########################################################################################################################################################
##########################################################################################################################################################

class SingleSessionCloseSet(BaseEvaluation):

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
                                                #Single Session Evaluatiom for Siamese Network(Close-set Scenario)
##########################################################################################################################################################
##########################################################################################################################################################


    def _siamese_training(self, data, y, siamese):

        """Train Siamese networks for close-set authentication.

        Parameters:
            data (numpy.ndarray): Input data for training.
            y (numpy.ndarray): Labels associated with the input data.
            siamese (SiameseNetwork): Siamese network object for training.

        Returns:
            tuple: Tuple containing dictionaries of Siamese network results.

        Description:
            This method trains Siamese networks for single-session authentication. It utilizes Stratified K-Fold cross-validation
            to split the data into training and test sets for model evaluation. The Siamese network is then trained on the training
            dataset and evaluated on the test dataset for each fold. The results are stored in dictionaries (dicr1, dicr2, dicr3),
            containing the outcomes of different evaluation metrics for each fold. The method returns a tuple containing these dictionaries.
        """
        count_cv=0
        dicr3={}
        dicr2={}
        dicr1={}
        skfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
        for train_index, test_index in skfold.split(data, y):
            x_train, x_test, y_train, y_test =data[train_index],data[test_index],y[train_index],y[test_index]
            scaler = StandardScaler()

            # Normalizing training and testing data using StandardScaler
            x_train = scaler.fit_transform(x_train.reshape((x_train.shape[0], -1))).reshape(x_train.shape)
            x_test = scaler.transform(x_test.reshape((x_test.shape[0], -1))).reshape(x_test.shape)
            if (siamese.user_siamese_path is None):

                # If the user siamese path is not provided, then we utilize the default siamese network
                model=siamese._siamese_embeddings(x_train.shape[1], x_train.shape[2])
            else:

                # If the user siamese path is provided, then we utilize the user siamese network
                model=siamese._user_siamese_embeddings(x_train.shape[1], x_train.shape[2])     
            embedding_network=model
            #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(siamese.batch_size)
            history = embedding_network.fit(train_dataset,
                                        workers=siamese.workers,
                                        epochs=siamese.EPOCHS,
                                        verbose=siamese.verbose)
            resutls1,resutls2,resutls3= CalculateSimilarity._close_set_identification(model, x_train, y_train, x_test, y_test) 
            dicr1[count_cv] = resutls1
            dicr2[count_cv] = resutls2
            dicr3.update(dict(resutls3))
            count_cv=count_cv+1
            tf.keras.backend.clear_session()
            del model, embedding_network, train_dataset, history
            gc.collect()
        return dicr3

    def deep_learning_method(self, X, dataset, metadata, key, features):

        """
        Perform authentication for single-session evaluation using Siamese networks.

        Parameters:
            dataset (Dataset): The dataset being used for evaluation.
            X (numpy.ndarray): The input features for evaluation.
            metadata (pandas.DataFrame): Metadata associated with the input features.
            key (str): Identifier for the specific evaluation pipeline.
            features (list): List of features used for evaluation.

        Returns:
            list: List of dictionaries containing evaluation results for each subject and session.

        This method utilizes Siamese networks for within-session evaluation in the context of biometric
        authentication. It involves creating and training Siamese networks to compute various authentication 
        metrics such as AUC, EER, FRR_1_FAR, and TPR for each subject and session combination in the dataset.
        The results are stored as dictionaries containing evaluation details like evaluation type, dataset, 
        pipeline used, subject, session, and corresponding evaluation metrics. The function returns a list of 
        dictionaries, each containing evaluation results for a particular subject and session.

        """
        # results_saving_path=os.path.join(
        #     dataset.dataset_path,
        #     "Results",
        #     "SiameseWithinSessionEvaluation"
        # )
        # if not os.path.exists(results_saving_path):
        #     os.makedirs(results_saving_path)

        metadata=metadata[metadata['event_id']=="Deviant"]
        metadata=self._valid_subject_samples(metadata)
        target_index=metadata['event_id'].index.tolist()
        data=X[target_index]
        y=np.array(metadata["subject"])
        results_close_set=[]
        for session in np.unique(metadata.session):
            ix = metadata.session == session
            #for name, clf in pipelines.items():
            siamese = features[0]
            le = LabelEncoder()
            X_=data[ix]
            y_=y[ix]
            close_dicr3=self._siamese_training(X_, y_, siamese)
            # close_set_path=os.path.join(results_saving_path,"close_set")
            # if not os.path.exists(close_set_path):
            #     os.makedirs(close_set_path)

            # with open(os.path.join(close_set_path, "d1_dicr1.pkl"), 'wb') as f:
            #     pickle.dump(close_dicr1, f)

            # with open(os.path.join(close_set_path, "d1_dicr2.pkl"), 'wb') as f:
            #     pickle.dump(close_dicr2, f)

            # with open(os.path.join(close_set_path, "d1_dicr3.pkl"), 'wb') as f:
            #     pickle.dump(close_dicr3, f)

            for sub in close_dicr3.keys():
                result=close_dicr3[sub]
                result=np.array(result)

                true_lables=np.array(result[:,1])
                true_lables=true_lables.astype(np.float64)
                predicted_scores=np.array(result[:,0])
                # print("predicted scores", predicted_scores)
                eer, frr_1_far=score._calculate_siamese_scores(true_lables, predicted_scores)
                res_close_set = {
                'evaluation': 'Single Session',
                    "eval Type": "Close Set",
                    "dataset": dataset.code,
                    "pipeline": key,
                    "subject": sub,
                    "session": session,
                    "frr_1_far": frr_1_far,
                    #"accuracy": mean_accuracy,
                   # "auc": auc,
                    "eer": eer,
                    #"tpr": inter_tpr,
                    #"std_auc": std_auc,
                    "n_samples": len(X_)  # not training sample
                    #"n_channels": data.columns.size
                    }
                results_close_set.append(res_close_set)

        return results_close_set
    
##########################################################################################################################################################
##########################################################################################################################################################
                                        #Single Session Evaluatiom for State-of-the-algorithms(Close-set Scenario)
##########################################################################################################################################################
##########################################################################################################################################################
    

    def _authenticate_single_subject_close_set(self, X, y, pipeline):

        """
        Perform authentication for a single subject in a close-set scenario.

        Parameters:
            X (numpy.ndarray): The input features for authentication.
            y (numpy.ndarray): The labels for the input features.
            pipeline (list): The authentication pipeline including classifier.

        Returns:
            dict: Tuple containing the average authentication scores across the k-fold
                  for each subject.

        This method evaluates the authentication performance for a single subject in a close-set scenario.
        It uses RepeatedStratifiedKFold cross-validation to split the data into training and test sets.
        The function normalizes the data, trains the model, predicts test set results, and calculates
        various authentication metrics (such as accuracy, AUC, EER, FPR, TPR) for each fold in the cross-validation.
        The average scores for accuracy, AUC, EER, and FRR_1_FAR are then computed and returned as a dictionary.
        """
        accuracy_list=[]
        auc_list=[]
        eer_list=[]
        fpr_list=[]
        tpr_list=[]
        fnr_list=[] 
        frr_1_far_list=[]

        # Defining the Stratified KFold
        skfold = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=42)

        # Intializing the classifier from the pipeline
        classifer=pipeline[-1]
        mean_fpr=np.linspace(0, 1, 100000)

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

            #print("length of features", len(df))
            #print("subject sample count", df['Subject'].value_counts())
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
        for subject in tqdm(np.unique(data.subject), desc=f"{key}-SingleSessionCloseSet"):
            df_subj=data.copy(deep=True)

            # Assign label 0 to all subjects
            df_subj['Label']=0

            # Updating the label to 1 for the subject being authenticated
            df_subj.loc[df_subj['subject'] == subject, 'Label'] = 1
            #print("'")
            for session in np.unique(df_subj.session):
                df_session= df_subj[df_subj.session==session]
                labels=np.array(df_session['Label'])
                X=np.array(df_session.drop(['Label','Event_id','subject','session'],axis=1))

                # if not self._valid_subject_samples(df_session):
                #     continue
                
                close_set_scores=self._authenticate_single_subject_close_set(X,labels, features)
                mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far=close_set_scores
                res_close_set = {
                # "time": duration / 5.0,  # 5 fold CV
                'evaluation': 'Single Session',
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
                #"n_samples": len(data)  # not training sample
                #"n_channels": data.columns.size
                    }
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
            if (key.upper()=='SIAMESE'):
                
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

        results=self._evaluate(dataset, pipelines)
        scenario="close_Set"
        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "SingleSessionEvaluation"
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
    
    def _valid_subject_session(self, df, subject, session):
    
        """
        Check if a subject has the required session for single-session evaluation.

        Parameters:
            - df: The data for the subject.
            - subject: The subject to be evaluated.
            - session: The session to be evaluated.

        Returns:
            - valid: A boolean indicating if the subject has the required session.
        """

        df_subject=df[df['subject']==subject]
        sessions=df_subject.session.values
        if (session not in sessions):
            return False
        
        else:
            return True
    
    def is_valid(self, dataset):

        """
        Check if a dataset is valid for evaluation.

        Parameters:
            - dataset: The dataset for evaluation.

        Returns:
            - True if the dataset is valid, otherwise False.
        """

        return True



    