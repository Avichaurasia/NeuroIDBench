from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
#from deeb.evaluation.evaluation import CloseSetEvaluation, OpenSetEvaluation

class Scores():
    
    def _calculate_average_scores(accuracy_list, tpr_list, eer_list, mean_fpr, auc_list, frr_1_far_list):
        """Calculating average scores like mean accuracy, mean auc, mean eer, mean tpr, tpr_upper, tpr_lower, std_auc
        for all k-folds"""

         # Averaging mean accuracy
        mean_accuracy=np.mean(accuracy_list, axis=0)

        # Averaging mean TPR
        mean_tpr=np.mean(tpr_list,axis=0)
        mean_tpr[-1] = 1.0

        # Average AUC
        mean_auc = metrics.auc(mean_fpr, mean_tpr)

        # Average standard deviation of TPR 
        std_tprs=np.std(tpr_list,axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tprs, 1)
        tprr_lower = np.maximum(mean_tpr - std_tprs, 0)

        # Standard deviation of AUC
        std_auc=np.std(auc_list, axis=0)

        # Average EER
        mean_eer=np.mean(eer_list, axis=0)

        # Average FRR at 1% FAR
        mean_frr_1_far=np.mean(frr_1_far_list, axis=0)
        return (mean_accuracy, mean_auc, mean_eer, mean_tpr, tprs_upper, tprr_lower, std_auc, mean_frr_1_far)
        
    def _calculate_scores(y_prob, y_test, mean_fpr):
        """Calculating scores like tpr, fpr, eer, inter_tpd for each k-fold"""
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)

        # Calculating False Negative Rate/False Reject Rate
        fnr=1-tpr
        inter_tpr=np.interp(mean_fpr, fpr, tpr)
        inter_tpr[0] = 0.0

        # Calculating Area under Curve for each k-fold
        auc=metrics.auc(fpr, tpr)
        
        # Calculating Equal Error Rate for each k-fold
        eer = brentq(lambda x : 1. - x - interp1d(mean_fpr, inter_tpr)(x), 0., 1.)
        #eer_other=brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        # Threshold for Equal Error Rate
        eer_thresh = interp1d(fpr, thresholds)(eer)

        # Calculating FRR at 1% FAR
        frr_1_far = interp1d(fpr, fnr)(0.01)
        return (auc, eer, eer_thresh, inter_tpr, tpr, fnr, frr_1_far)





