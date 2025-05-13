import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

filepath = ''
os.makedirs(os.path.join(filepath, 'results/percolation_threshold'), exist_ok=True)

# cleaning
input_file = os.path.join(filepath,'results/gower_real_real.pkl')
distance_matrix_df = pd.read_pickle(input_file)
distance_matrix_df = distance_matrix_df.mask(np.triu(np.ones(distance_matrix_df.shape)).astype(bool)).transpose().stack()
distance_matrix_df.index.rename(['Obj1', 'Obj2'], inplace=True)
distance_matrix_df = distance_matrix_df.to_frame('Dist').reset_index()

# threshold curve
threshold_curve = distance_matrix_df.groupby(['Dist'])['Dist'].count()
threshold_curve = threshold_curve.to_frame('connections').reset_index()
threshold_curve['connections'] = threshold_curve['connections'].cumsum()
threshold_curve.to_csv(os.path.join(filepath,'results/percolation_threshold/threshold_curve_gower.csv'))

## INFLECTION POINT
# Define the sigmoidal (logistic) function
def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Define your data points (x and y values)
x_data = threshold_curve['Dist'].values
y_data = threshold_curve['connections'].values

# normalize data to avoid runtime error
y_max = np.max(y_data)
y_min = np.min(y_data)
y_data_normalized = (y_data - y_min) / (y_max - y_min)

# Fit the sigmoidal curve to the data
params, covariance = curve_fit(sigmoid, x_data, y_data_normalized)

# Extract the fitted parameters
L, k, x0 = params

# Find the inflection point of the sigmoid function
inflection_point = x0

# y_predicted = sigmoid(x0, L, k, x0)
# y_predicted * (y_max - y_min) + y_min

## INTERSECTION POINT
def sigmoid_derivative(x, L, k, x0):
    return L * k * np.exp(k * (x - x0)) / ((1 + np.exp(k * (x - x0))) ** 2)

# Calculate the slope (m) of the tangent line at the inflection point
m = sigmoid_derivative(inflection_point, L, k, x0)

# Calculate the y-intercept (b) of the tangent line
# tangent line: y = m * x + b, --> b = y - m * x
b = sigmoid(inflection_point, L, k, x0) - m * inflection_point

# Find the x where the tangent line intersects with y = 0
# 0 = m * x + b --> x = -b / m
intersection_x = -b / m

# # The intersection point is (intersection_x, 0)
# print("Intersection Point: ({}, 0)".format(intersection_x))

# threshold_to_cytoscape
cytoscape_df = distance_matrix_df[(distance_matrix_df["Dist"] >= 0) & (distance_matrix_df["Dist"] <= intersection_x)] 
print(str(len(pd.concat([cytoscape_df['Obj1'], cytoscape_df['Obj2']]).unique()))+ ' nodes (threshold)')
cytoscape_df.to_csv(os.path.join(filepath,'results/percolation_threshold/cytoscape_gower'+str(round(intersection_x,2))+'.csv', index=False))

# plot
plt.figure(figsize=(16,10))
plt.scatter(threshold_curve['Dist'],threshold_curve['connections'])       
max_connections = round(max(threshold_curve['connections']), -8)
plt.vlines(x=[x0], ymin=0, ymax=(max_connections+0.5e8), colors='green', ls='--', lw=2, label=f'Percolation Threshold: {x0:.2f}')
plt.vlines(x=[intersection_x], ymin=0, ymax=(max_connections+0.5e8), colors='red', ls='--', lw=2, label=f'Intersection Point: {intersection_x:.2f}')
plt.ticklabel_format(style='plain')
plt.title("Threshold Curve - Gower")
plt.xlabel("Distance")
plt.ylabel("Num Connections")
plt.legend(loc='upper right')
plt.savefig(os.path.join(filepath,'results/percolation_threshold/threshold_curve_gower.png'))
plt.close()