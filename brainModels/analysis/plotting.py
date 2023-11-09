from brainModels.Evaluation.base import BaseEvaluation
from brainModels.analysis.results import Results as res
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')
import logging
import os
from mne import get_config, set_config
from mne.datasets.utils import _get_path
import seaborn as sns

log = logging.getLogger(__name__)

class Plots():

    def __init__(self,
    plot_path=None,
    ):
        if plot_path is None:
            mne_data_path = get_config("MNE_DATA")
            if mne_data_path is None:
                set_config("MNE_DATA", os.path.join(os.path.expanduser("~"), "mne_data"))
                mne_data_path = get_config("MNE_DATA")
            self.plot_path = os.path.join(mne_data_path, "Deeb_plots")
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
        else:
            self.plot_path = plot_path
        
    def _plot_erp():
        return 0
    
    def _roc_curve_single_dataset(self, data=None, evaluation_type=None, dataset=None):
        #print("Plotting roc curve for single dataset", self.plot_path)
        file_path=os.path.join(self.plot_path, "Single_dataset_Roc_curves")
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        grouped_df = data.groupby(['dataset', 'pipeline']).agg({
            'accuracy': 'mean',
            'auc': 'mean',
            'eer': 'mean',
            'tpr': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
            'tprs_lower': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
            'tprs_upper': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
            'std_auc': 'mean',
            'n_samples': 'mean'
             }).reset_index()
        
        fig, ax = plt.subplots(figsize=(9,6))
        for i in range(len(grouped_df)):
            name = grouped_df['pipeline'][i]
            fpr=np.linspace(0, 1, 100)
            auc = grouped_df['auc'][i]
            std_auc=grouped_df['std_auc'][i]
            #std_auc=np.std(grouped_df['auc'][i])
            tpr = grouped_df['tpr'][i]
            
            # Plot the ROC curve
            ax.plot(fpr, tpr,label=name+" "+r'(AUC = %0.3f $\pm$ %0.3f)' % (auc, std_auc),
                    lw=2, alpha=.8)
            # Add labels and legend
            plt.title("ROC Curve: "+evaluation_type,fontsize=12)
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.major.formatter._useMathText = True
            ax.legend(frameon=True, loc='best',ncol=1, handlelength=2, framealpha=1, edgecolor="0.8", fancybox=False)
            plt.tight_layout()
            plt.grid(True, ls="--", lw=0.8)
        plt.plot([0, 1], [0, 1], "k--", color='b',label="chance level (AUC = 0.5)")
        fname=os.path.join(file_path, dataset.code+"_"+evaluation_type+'.pdf')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        
    def _roc_curve_multiple_datasets(self, data=None, evaluation_type=None, datasets=None):
        file_path=os.path.join(self.plot_path, "roc_curve_multiple_datasets")
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        n_datasets = len(datasets)
        n_rows = int(np.ceil(n_datasets / 2))  # Round up to the nearest integer
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
        # Flatten the axes array
        axes = axes.flatten()
        # Iterate over each dataset and plot the ROC curve
        for i, (df, dataset) in enumerate(zip(data, datasets)):

            # Select the current subplot
            ax = axes[i]

            # Group the data by pipeline and aggregate the scores
            grouped_df = df.groupby(['pipeline']).agg({
                'accuracy': 'mean',
                'auc': 'mean',
                'eer': 'mean',
                'tpr': lambda x: np.mean(np.vstack(x), axis=0),
                'tprs_lower': lambda x: np.mean(np.vstack(x), axis=0),
                'tprs_upper': lambda x: np.mean(np.vstack(x), axis=0),
                'std_auc': 'mean',
                'n_samples': 'mean'
            }).reset_index()

            # Plot the ROC curve
            for j in range(len(grouped_df)):
                name = grouped_df['pipeline'][j]
                fpr = np.linspace(0, 1, 100)
                auc = grouped_df['auc'][j]
                std_auc = grouped_df['std_auc'][j]
                tpr = grouped_df['tpr'][j]
                ax.plot(fpr, tpr, label=name + " " + r'(AUC = %0.3f $\pm$ %0.3f)' % (auc, std_auc),
                        lw=2, alpha=.8)
                
                # Add labels and legend
                ax.set_title(dataset.code, fontsize=12)
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.major.formatter._useMathText = True
                ax.legend(frameon=True, loc='best', ncol=1, handlelength=2, framealpha=1, edgecolor="0.8", fancybox=False)
                ax.grid(True, ls="--", lw=0.8)
                ax.plot([0, 1], [0, 1], "k--", color='b', label="chance level (AUC = 0.5)")
                plt.tight_layout()
            
        fname = os.path.join(file_path, 'roc_curve_multiple_datasets.pdf')
        plt.show()
        #plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _det_curve_single_dataset():
        return 0
    
    def _det_curve_multiple_datasets():
        return 0
    
    def _eer_single_dataset(self, data=None, evaluation_type=None, dataset=None):
        file_path=os.path.join(self.plot_path, "Single_dataset_EER_curves")
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        grouped_df = data.groupby(['dataset', 'pipeline']).agg({
            'accuracy': 'mean',
            'auc': 'mean',
            'eer': 'mean',
            'tpr': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
            'tprs_lower': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
            'tprs_upper': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
            'std_auc':'mean',
            'n_samples': 'mean'
             }).reset_index()
        
        # Convert eer to %eer for plotting
        grouped_df['eer'] = grouped_df['eer']*100

        # Plotting the bar graph for %eer across pipelines with %eer on x-axis, pipelines on y-axis and %eer as bar height with maximum and minimum values
        fig, ax = plt.subplots(figsize=(9,6))
        for i in range(len(grouped_df)):
            name = grouped_df['pipeline'][i]
            eer = grouped_df['eer'][i]
            # Plot the EER curve
            ax.barh(name, eer, color='b', lw=2, alpha=.8)
            # Add labels and legend
            plt.title("EER Curve: "+evaluation_type,fontsize=12)
            plt.xlabel('Percentage EER', fontsize=12)
            plt.ylabel('Pipelines', fontsize=12)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.major.formatter._useMathText = True
            ax.legend(frameon=True, loc='best',ncol=1, handlelength=2, framealpha=1, edgecolor="0.8", fancybox=False)
            plt.tight_layout()
            plt.grid(True, ls="--", lw=0.8)
        fname=os.path.join(file_path, dataset.code+"_"+evaluation_type+'.pdf')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    def _plot_eer_multiple_datasets():
        return 0
    
    def _eer_graph_single_dataset_across_evaluations(self, data=None):
        eer_path=os.path.join(self.plot_path, "EER_across_evaluations")
        if not os.path.exists(eer_path):
            os.makedirs(eer_path)

        # Raise value error if evaluation_type is not a list of length 2
        if not isinstance(data, list) or len(data)>1:
            raise ValueError("data must be a list")
        
        else:
            # Get the eer values for the two evaluation types
            close_set_df = data[0].groupby(['dataset', 'pipeline']).agg({
                'accuracy': 'mean',
                'auc': 'mean',
                'eer': 'mean',
                'tpr': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
                'tprs_lower': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
                'tprs_upper': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
                'std_auc':'mean',
                'n_samples': 'mean'
                }).reset_index()
            
            open_set_df = data[1].groupby(['dataset', 'pipeline']).agg({
                'accuracy': 'mean',
                'auc': 'mean',
                'eer': 'mean',
                'tpr': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
                'tprs_lower': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
                'tprs_upper': lambda x: np.mean(np.vstack(x), axis=0),  # average across numpy arrays
                'std_auc':'mean',
                'n_samples': 'mean'
                }).reset_index()
            
            # Convert eer to %eer for plotting
            close_set_df['eer'] = close_set_df['eer']*100
            open_set_df['eer'] = open_set_df['eer']*100   

            # create a list of 12 pipelines for both open-set and close-set
            pipelines = list(close_set_df['pipeline'].unique()[:12])

            # get the EER values for the open-set pipelines
            eer_open = open_set_df['eer'].iloc[:12]
            dataset_name=close_set_df['dataset'].unique()[0]

            # get the EER values for the close-set pipelines
            eer_close = close_set_df['eer'].iloc[:12]

            # set the width of the bars
            bar_width = 0.35

            # set the position of the bars on the x-axis
            r1 = np.arange(len(pipelines))
            r2 = [x + bar_width for x in r1]

            # plot the bars for open-set and close-set
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(r1, eer_open, color='b', width=bar_width, edgecolor='white', label='Open-set')
            ax.bar(r2, eer_close, color='g', width=bar_width, edgecolor='white', label='Close-set')

            # add labels and title
            plt.title("Percentage EER: "+dataset_name, fontsize=12)
            plt.xlabel('Pipelines', fontsize=12)
            plt.xticks([r + bar_width / 2 for r in range(len(pipelines))], pipelines, fontsize=10, rotation=45, ha='right')
            plt.ylabel('Percentage EER', fontsize=12)

            # add legend
            plt.legend()
            fname=os.path.join(eer_path, dataset_name+"_"+"_EER_"+".pdf")
            plt.savefig(fname, dpi=300, bbox_inches='tight')
        
    
    def _eer_graph_across_datasets(self, data=None, evaluation_type=None, dataset=None):
        return 0
        #return 0
    
