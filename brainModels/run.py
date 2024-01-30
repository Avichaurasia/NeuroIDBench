import numpy as np
import sys
sys.path.append("../")
from benchmark import benchmark

if __name__ == "__main__":
    #print("Running benchmark")
    result=benchmark()
    #print
    grouped_df=result.groupby(['evaluation','pipeline', 'eval Type','dataset']).agg({
                "subject": 'nunique',
                'auc': 'mean',
                'eer': lambda x: f'{np.mean(x)*100:.3f} ± {np.std(x)*100:.3f}',
                'frr_1_far': lambda x: f'{np.mean(x)*100:.3f}'
            }).reset_index()
    grouped_df.rename(columns={'eval Type':'Scenario', 'subject':'Subjects'}, inplace=True)
    print(grouped_df)
