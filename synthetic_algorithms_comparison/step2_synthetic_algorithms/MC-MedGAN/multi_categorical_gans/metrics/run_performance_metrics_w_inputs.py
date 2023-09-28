from __future__ import division
from __future__ import print_function

import sys
#path_to_mc-medgan: synthetic_algorithms_comparison/step2_synthetic_algorithms/MC-MedGAN
sys.path.append("/path_to_mc-medgan")

import numpy as np
import pandas as pd
from multi_categorical_gans.metrics import performance_metrics
import types
import argparse

def metrics(data_a_path, data_b_path, result_path):
    data_a = pd.read_csv(data_a_path)
    data_a = data_a.drop(columns = ["PatientID", "Unnamed: 0"]) 
    print(data_a.shape)

    data_b = pd.read_csv(data_b_path)    
    data_b = data_b.drop(columns = ["PatientID"])
    print(data_b.shape)

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
                df.to_csv(result_path)         



def main():
    options_parser = argparse.ArgumentParser(description="Calculate performance metrics for synthetic data.")

    options_parser.add_argument("data_a", type=str, help="Input SEER data in text format.")
    options_parser.add_argument("data_b", type=str, help="Input sampling in csv format.")
    options_parser.add_argument("results", type=str, help="output path.")

    options = options_parser.parse_args()

    metrics(options.data_a, options.data_b, options.results)



if __name__ == "__main__":
    main()
