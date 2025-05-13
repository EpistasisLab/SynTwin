# Code to calculate distance matrices between samples in the holdout dataset.
# The distance values are used to create the patient network and calculate the percolation threshold.

import pandas as pd
import numpy as np
from cdist_gower import cdist_gower
import os 

filepath = ''
real_data = pd.read_csv(os.path.join(filepath,'data/synthetic.csv'))
real_ids = real_data['IID']
real_data.drop(columns = ['IID', 'outcome'], inplace=True)

# Gower Distance
cat_features = [False] * 87
gower_matrix = cdist_gower(real_data.iloc[:,0:], cat_features = cat_features)
gower_df = pd.DataFrame(gower_matrix, index=real_ids, columns=real_ids)

gower_df.to_pickle(os.path.join(filepath, "results/gower_real_real.pkl"))