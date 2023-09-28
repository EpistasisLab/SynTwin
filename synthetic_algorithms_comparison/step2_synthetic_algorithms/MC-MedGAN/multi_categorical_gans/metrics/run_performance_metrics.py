import sys
#path_to_mc-medgan: synthetic_algorithms_comparison/step2_synthetic_algorithms/MC-MedGAN
sys.path.append("/path_to_mc-medgan")

import numpy as np
import pandas as pd
from multi_categorical_gans.metrics import performance_metrics
import types

data_a = pd.read_csv('/path_to_mc-medgan/data/seerbreast/encoded_data.txt')
data_a = data_a.iloc[: , 2:]

data_b = pd.read_csv('/path_to_mc-medgan/samples/mc-medgan/seerbreast/sample.csv')
data_b = data_b.drop(columns = ["PatientID"])

metrics = ['kl_divergence',
               'cross_classification',
                'cca_accuracy',
                'cluster_measure',
                'pairwise_correlation_difference',
               'coverage',
               'membership_disclosure',
               'percentage_revealed',
               'attribute_disclosure'
    ]

# get list of available metrics
metric_func = {a: performance_metrics.__dict__.get(a)
               for a in dir(performance_metrics)
               if isinstance(performance_metrics.__dict__.get(a),
                             types.FunctionType)}

# initialize results storage structure
results = {}
for met in metrics:
    results[met] = list()

# dict to save performance metrics for the t-th task
for met in metrics:
    # metric m for method m in the r-th run
    m_m_r = metric_func[met](data_a=data_a, data_b=data_b)
    results[met].append(m_m_r)


result_contents = list()
method ='MC-MedGAN'

# for each metric
for k in results.keys():
    # for each sampled synthetic dataset
    for i in range(len(results[k])):
        # iterate over metrics for k-th sampled dataset
        type = 'Multiple' if len(results[k][i].keys()) > 1 else 'Single'
        for var in results[k][i].keys():
            result_contents.append([method, k, var, i+1, type, results[k][i][var]])

            # store result_contents list into a dataframe for easier manipulation
            column_names = ['Method', 'Metric', 'Variable', 'Run', 'Type', 'Value']
            df = pd.DataFrame(result_contents, columns=column_names)
            df.to_csv('/path_to_mc-medgan/samples/mc-medgan/seerbreast/utility_metrics.csv')