import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import h5py
import json
#from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')
import os
import seaborn as sns
import random
import os
import pickle
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
#import matplotlib.pyplot as plt

#os.chdir('/Users/avinashkumarchaurasia/mne_data/MNE-erpcoren400-data/v1/resources/29xpq/providers/osfstorage/Results')

brain=pd.read_csv('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/Results/ROC_Curves/brainInbaders15.csv')
erpN400=pd.read_csv('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/Results/ROC_Curves/ERPCOREN400.csv')
erpP300=pd.read_csv('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/Results/ROC_Curves/ERPCOREP300.csv')
cogbci=pd.read_csv('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/Results/ROC_Curves/COGBCI.csv')
#huebner=pd.read_csv('/scratch/hpc-prf-bbam/avinashk/Brain-Models/examples/Lee2019_dataset/AUC_datasets/Huebner2017.csv')
sosulski=pd.read_csv('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/Results/ROC_Curves/Sosulski.csv')
#won2022=pd.read_csv('/scratch/hpc-prf-bbam/avinashk/Brain-Models/examples/Lee2019_dataset/AUC_datasets/Won2022.csv')
mantegna2019=pd.read_csv('/Users/avinashkumarchaurasia/Desktop/project/BrainModels/Results/ROC_Curves/Mantegna.csv')

# def _roc_curves():
#     return 0

# # main function
# if __name__ == '__main__':
#     _roc_curves()

ls=[brain, erpN400, erpP300, cogbci, sosulski , mantegna2019]
df=pd.concat(ls, axis=0)

df.drop(columns=['evaluation', 'Unnamed: 0'], inplace=True)
df['pipeline'] = df['pipeline'].apply(lambda x: x.split('+')[-1])
df['pipeline'].replace({'siamese':'TNN'}, inplace=True)
df['dataset'].replace({'Visual Speller LLP': "Huebner2017", 'Spot Pilot P300 dataset':'Sosulski2019'}, inplace=True)


# for i, val in enumerate(df['dataset'].tolist()):
#     print(type(val))
# #print(df.dtypes)
# #df['tpr'] = df['tpr'].astype(np.ndarray)
# grouped_df=df.groupby(['pipeline', 'eval Type', 'dataset']).agg({
#                 # 'accuracy': 'mean',
#                 'tpr': lambda x: np.mean(np.vstack(x), axis=0),
#                 'auc': lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}',
#                 'eer': 'mean',
#                 'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
#             }).reset_index()
    
# Convert the 'dataset' column to NumPy arrays
df['tpr'] = df['tpr'].apply(lambda x: np.array(x).tolist())

print("avinash")
# Apply aggregation
grouped_df = df.groupby(['pipeline', 'eval Type', 'dataset']).agg({
                # 'accuracy': 'mean',
                'tpr': 'mean',
                'auc': lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}',
                # 'eer': 'mean',
                # 'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()

# grouped_df = df.groupby(['pipeline', 'eval Type', 'dataset']).agg({
#     'tpr': lambda x: np.mean([list(arr) for arr in x], axis=0),
#     'auc': lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}',
#     'eer': 'mean',
#     'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
# }).reset_index()

# Convert the 'tpr' column to a list of arrays
# df['tpr'] = df['tpr'].apply(lambda x: [np.array(arr) for arr in x])

# # Group by and aggregate
# grouped_df = df.groupby(['pipeline', 'eval Type', 'dataset']).agg({
#     'tpr': lambda x: np.mean(np.vstack(x), axis=0),
#     'auc': lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}',
#     'eer': 'mean',
#     'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
# }).reset_index()


# # Convert the 'tpr' column to lists
# print(type(df['tpr'][0].values))
# df['tpr'] = df['tpr'].apply(lambda x: np.float(x).tolist())

# # Apply aggregation
# grouped_df = df.groupby(['pipeline', 'eval Type', 'dataset']).agg({
#                 # 'accuracy': 'mean',
#                 'tpr': lambda x: np.mean(x, axis=0),
#                 'auc': lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}',
#                 'eer': 'mean',
#                 'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
#             }).reset_index()

print(grouped_df)

