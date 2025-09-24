# Code to calculate distance matrices between samples in the holdout dataset.
# The distance values are used to create the patient network and calculate the percolation threshold.

import pandas as pd
import numpy as np
from cdist_gower import cdist_gower
import os 

def step2_calculate_distance(filepath):
	print("Running step2_calculate_distance")
	real_data = pd.read_csv(os.path.join(filepath,'gene_scores_test_nonad.csv'))
	real_ids = real_data['IID']
	real_data.drop(columns = ['IID', 'outcome'], inplace=True)

	# Gower Distance
	num_cols = real_data.shape[1]
	print(f"num_cols: {num_cols}")
	cat_features = [False] * num_cols

	num_max = np.ones(num_cols)
	num_ranges = np.zeros(num_cols)
	for idx, col_name in enumerate(real_data.columns):
		col_array = real_data[col_name].astype(np.float32).values
		max_val = np.nanmax(col_array)
		min_val = np.nanmin(col_array)

		if np.isnan(max_val):
		    max_val = 0.0
		if np.isnan(min_val):
		    min_val = 0.0

		num_max[idx] = max_val
		num_ranges[idx] = np.abs(1 - min_val / max_val) if max_val != 0 else 0.0        


	gower_matrix = cdist_gower(real_data.iloc[:,0:], cat_features = cat_features, num_max=num_max, num_ranges=num_ranges)
	gower_df = pd.DataFrame(gower_matrix, index=real_ids, columns=real_ids)

	gower_df.to_pickle(os.path.join(filepath, "results/gower_real_real.pkl"))

	print("step2_calculate_distance completed")