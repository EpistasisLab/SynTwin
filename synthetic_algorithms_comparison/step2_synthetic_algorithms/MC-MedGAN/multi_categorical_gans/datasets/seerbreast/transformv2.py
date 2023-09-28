from __future__ import print_function

import argparse
import csv
import json

import numpy as np
import pandas as pd
import os.path as path

from scipy.sparse import csr_matrix, save_npz


def seer_transform(input_path, output_path, metadata_path, datadim_path):
    input_file = open(input_path, "r")
    reader = csv.DictReader(input_file)

    variables = sorted(reader.fieldnames)
    variables.remove("") # remove index
    variables.remove("PatientID")

    values_by_variable = {}
    for variable in variables:
        values_by_variable[variable] = set()

    print("Counting values...")
    for row_number, row in enumerate(reader):
        if row_number % 10000 == 0:
            print("{:d} rows read".format(row_number))

        for variable in variables:
            value = row[variable]
            values_by_variable[variable].add(value)

    print("Saving metadata...")

    feature_number = 0
    index_to_value = []
    value_to_index = {}
    variable_sizes = []
    for variable in variables:
        values = sorted(values_by_variable[variable])
        variable_sizes.append(len(values))
        value_to_index[variable] = {}
        for value in values:
            index_to_value.append((variable, value))
            value_to_index[variable][value] = feature_number
            feature_number += 1

    num_samples = row_number + 1
    num_features = feature_number

    metadata = {
        "variables": variables,
        "variable_sizes": variable_sizes,
        "index_to_value": index_to_value,
        "value_to_index": value_to_index,
        "num_samples": num_samples,
        "num_features": num_features
    }

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)

    input_file.close()

    input_file = open(input_path, "r")
    reader = csv.DictReader(input_file)

    ones = []
    rows = []
    cols = []

    print("Writing...")

    for row_number, row in enumerate(reader):
        if row_number % 10000 == 0:
            print("{:d} rows read".format(row_number))

        for variable in variables:
            value = row[variable]
            feature_number = value_to_index[variable][value]

            ones.append(1)
            rows.append(row_number)
            cols.append(feature_number)

    output = csr_matrix((ones, (rows, cols)), shape=(num_samples, num_features), dtype=np.uint8)

    save_npz(output_path, output)

    print(num_samples, num_features)
    
    data_name = input_path.split('/')[-1].split('.')[0]
    list_row = [data_name, num_samples, num_features]

    if path.exists(datadim_path):
        df = pd.read_csv(datadim_path)
        df.loc[len(df.index)] = list_row
    else:
        column_names = ['Dataset', 'NumSamples', 'NumFeatures']
        df = pd.DataFrame([list_row], columns=column_names)
    
    df.to_csv(datadim_path, index=False) 

    #print("Done.")

    input_file.close()


def main():
    options_parser = argparse.ArgumentParser(
        description="Transform the SEER text data into feature matrix."
                    + " Dataset: "
    )

    options_parser.add_argument("input", type=str, help="Input SEER data in text format.")
    options_parser.add_argument("output", type=str, help="Output features in sparse scipy matrix format.")
    options_parser.add_argument("metadata", type=str, help="Metadata in json format.")
    options_parser.add_argument("datadim_path", type=str, help="Output data dimentions after transforming.")
    

    options = options_parser.parse_args()

    seer_transform(options.input, options.output, options.metadata, options.datadim_path)


if __name__ == "__main__":
    main()
