from ..analysis.results import Results as res
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')
import logging
import os
from mne import get_config, set_config
import pandas as pd
from mne.datasets.utils import _get_path
import seaborn as sns

log = logging.getLogger(__name__)

class Plots():

    """
    A class for generating and saving various plots and visualizations.

    This class provides methods for creating and saving different types of plots, such as ROC curves, EER curves,
    and other visualizations related to evaluations. It also handles the organization of plot files in specified
    directories.

    Attributes:
    - plot_path: The path where plot files are saved.

    Methods:
    - _plot_erp():
      A method for plotting event-related potentials (ERP). [This method currently lacks a specific implementation.]

    - _roc_curve_single_dataset(data=None, evaluation_type=None, dataset=None):
      Generate and save ROC curves for a single dataset and evaluation type.

    - _eer_single_dataset(data=None, evaluation_type=None, dataset=None):
      Generate and save EER (Equal Error Rate) curves for a single dataset and evaluation type.

    - _eer_graph_single_dataset_across_evaluations(data=None):
      Generate and save EER graphs for a single dataset across multiple evaluation types.

      """

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
        
    
    #def _roc_curves(self, data=None, evaluation_type=None, dataset=None):
    
    def _plot_roc(self, data):

        """
        Generate and save ROC curves for a single dataset and evaluation type.

        Parameters:
        - data: A DataFrame containing results data.
        - evaluation_type: The type of evaluation (e.g., "close_set", "open_set").
        - dataset: The dataset to be evaluated.

        Returns:
        - None
        
        """

        # Assert if data is not a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a DataFrame")
        
        grouped_df = data.groupby(['evaluation','eval Type','dataset', 'pipeline']).agg({
           'tpr': lambda x: np.mean(np.vstack(x), axis=0),
            'auc': lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}',
             }).reset_index()
        
        # Get the unique pipelines names
        grouped_df.rename(columns={'eval Type':'Scenario'}, inplace=True)
        grouped_df['pipeline'] = grouped_df['pipeline'].apply(lambda x: x.split('+')[-1])        
        evaluation_type = grouped_df['evaluation'].values[0]     
        #fig, ax = plt.subplots(figsize=(9,6))

        
        for scenario in grouped_df['Scenario'].unique():
            fig, ax = plt.subplots(figsize=(6,5))
            scenario_df=grouped_df[grouped_df['Scenario']==scenario]
            scenario_df=scenario_df.reset_index()
            #display(grouped_df)
            for i in range(len(scenario_df)):
                name = scenario_df['pipeline'][i]
                fpr=np.linspace(0, 1, 100000)
                auc = scenario_df['auc'][i]
                dataset=scenario_df['dataset'][i]
                tpr = scenario_df['tpr'][i]

                # Plot the ROC curve
                ax.plot(fpr, tpr,label=name+" "+"(AUC = "+auc+")", 
                        lw=2, alpha=.6)
                # Add labels and legend
                plt.title("ROC: "+dataset.replace(" ","").upper()+" ("+scenario+")",fontsize=14)
                plt.xlabel('FMR', fontsize=14)
                plt.ylabel('1-FNMR', fontsize=14)
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.major.formatter._useMathText = True
                ax.legend(frameon=True, loc='best',ncol=1, handlelength=3, framealpha=1, edgecolor="0.8", fancybox=False)
                plt.tight_layout()
                plt.grid(True, ls="--", lw=0.8)
            plt.plot([0, 1], [0, 1], "k--", color='b',label="chance level (AUC = 0.5)")
            plt.show()   
        
    def _plot_eer(self, data):

        """
        Generate the EER (Equal Error Rate) curves for a single dataset.

        parameters:
        - data: Data containing evaluation results.
        - evaluation_type: A string specifying the type of evaluation (e.g., 'known attacker' or 'unknown attacker').
        - dataset: The name of the dataset for labeling the plot.

        Returns:
        - None
        """

        grouped_df = data.groupby(['evaluation','eval Type','dataset', 'pipeline']).agg({
            'eer': 'mean'
             }).reset_index()
        
        # Convert eer to %eer for plotting
        grouped_df['eer'] = grouped_df['eer']*100

        # Extracting just the algorithm name
        grouped_df['pipeline'] = grouped_df['pipeline'].apply(lambda x: x.split('+')[-1])

        # Pivot the DataFrame to plot bars for close-set and open-set EER
        pivot_df = grouped_df.pivot(index='pipeline', columns='eval Type', values='eer')

        # Plotting the bar chart
        ax = pivot_df.plot(kind='bar', figsize=(10, 6))

        # Adding values on top of bars
        # Adding values on top of bars with adjusted text size
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() * 1.010, p.get_height() * 1.010), fontsize=8)
            
        plt.xlabel('Algorithm')
        plt.ylabel('%EER')
        plt.title('EER: dataset '+grouped_df['dataset'][0].replace(" ","").upper())
        plt.legend(title='Scenario')
        plt.grid(True, ls="--", lw=0.8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _plot_eer_across_datasets(self, data):
        """
        Generate the EER graphs for multiple datasets.

        Parameters:
        - data: Data containing evaluation results of multiple datasets.
        - evaluation_type: A string specifying the type of evaluation (e.g., 'known attacker' or 'unknown attacker').

        Returns:
        - None
        """

        grouped_df = data.groupby(['evaluation','eval Type','dataset', 'pipeline']).agg({
            'eer': 'mean'
             }).reset_index()
        
        # Write code to generate EER graphs for multiple datasets
        grouped_df['eer'] = grouped_df['eer']*100
        grouped_df['pipeline'] = grouped_df['pipeline'].apply(lambda x: x.split('+')[-1])
        pivot_df = grouped_df.pivot(index='pipeline', columns='dataset', values='eer')
        ax = pivot_df.plot(kind='bar', figsize=(10, 6))
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() * 1.010, p.get_height() * 1.010), fontsize=8)
        plt.xlabel('Algorithm')
        plt.ylabel('%EER')
        plt.title('EER: across datasets')
        plt.legend(title='Dataset')
        plt.grid(True, ls="--", lw=0.8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # def _plot_roc_across_datasets(self, data):

    #     """
    #     Generate and save ROC curves for multiple datasets.

    #     Parameters:
    #     - data: Data containing evaluation results of multiple datasets.
    #     - evaluation_type: A string specifying the type of evaluation (e.g., 'known attacker' or 'unknown attacker').
    #     - dataset: The name of the dataset for labeling the plot.

    #     Returns:
    #     - None
    #     """

        



       
    
    
        
    
    
    
