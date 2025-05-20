import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_performance(y_true, y_predict):
    y_predict_prob = y_predict
    y_predict = np.round(y_predict)
    Precision = round(precision_score(y_true=y_true, y_pred=y_predict), 4) 
    Recall = round(recall_score(y_true=y_true, y_pred=y_predict), 4)
    Accuracy = round(accuracy_score(y_true=y_true, y_pred=y_predict), 4)
    BalancedAccuracy = round(balanced_accuracy_score(y_true=y_true, y_pred=y_predict), 4)
    F1 = round(f1_score(y_true=y_true, y_pred=y_predict), 4)

    num_class = np.unique(np.array(y_true))
    if len(np.unique(np.array(y_true))) == 1:
        AUROC = np.nan
    else:
        AUROC = round(roc_auc_score(y_true=y_true, y_score=y_predict_prob), 4)
    CM = confusion_matrix(y_true=y_true, y_pred=y_predict)
    performance_list = [Precision, Recall, Accuracy, BalancedAccuracy, F1, AUROC, CM]
    return performance_list


def predict_outcome(real_data, y, feature, real_node, id_col, predict_col):
    #count and std
    count=len(y)
    std = y.std()     
    dead = y.sum()
    ratio = dead / len(y)

    #predict by mean
    pred_label_mean = y.mean() 

    #predict by mode
    pred_label_mode = y.mode()[0]  

    
    #predict by knn
    X_train = feature
    y_train = y

    X_test = real_data[real_data[id_col]==real_node].drop(columns={id_col, predict_col}).values 
    y_test = real_data[real_data[id_col]==real_node][predict_col].values
    
    knn = KNeighborsClassifier(n_neighbors=count)
    knn.fit(X_train, y_train)
    
    pred_label_knn = knn.predict(X_test)[0]
    
    #true label
    true_label = real_data[real_data[id_col]==real_node][predict_col].item()
    
    
    return count, std, dead, ratio, pred_label_mean, pred_label_mode, pred_label_knn, true_label


def bootstrap_resample_90perc_performance(all_predict, method, datasets, predict_functions, output_path):
    pt_id = pd.DataFrame(all_predict['Real_Node'].unique())#.tolist()
    n = round(0.9 * len(pt_id))    
    total_patient = len(pt_id)

    performances =[]
    
    for seed in tqdm(range(1000)): 
        for dataset in datasets:
            all_df = all_predict[(all_predict['Dataset']==dataset)]

            for predict_function in predict_functions:

                cols_to_select = [col for col in all_predict.columns if predict_function in col]
                cols_to_select.extend(['Real_VitalStatus', 'Real_Node'])
                df = all_df[cols_to_select]
                df = df.dropna(subset=[predict_function])
                sub_df = resample(df, n_samples=n, random_state=seed) 

                num_patient = len(sub_df['Real_Node'])
                num_unique_patient = len(sub_df['Real_Node'].unique())

                y_true = sub_df['Real_VitalStatus']
                y_pred = sub_df[predict_function]

                performance_list = compute_performance(y_true=y_true, y_predict=y_pred)
                performance = [method, seed, total_patient, num_patient, num_unique_patient, dataset, predict_function]
                performance.extend(performance_list)
                performances.append(performance) 

    cols = ['Method', 'SamplingRandomState', 'TotalNumPatient', 'SamplingNumPatient', 'SamplingNumUniquePatient', 'Dataset', 'PredictionFunction', 'Precision', 'Recall', 'Accuracy', 'BalancedAccuracy','F1', 'AUROC', 'CM']
    performances_df= pd.DataFrame(performances, columns=cols)
    os.makedirs(output_path+'90perc bootstrap performance', exist_ok=True)
    performances_df.to_csv(output_path+'90perc bootstrap performance/'+method+'_90perc_performances.csv',index=False)
    
    return performances_df


def show_metric_dist(performances_df, method, predict_function, dataset, output_path):
    os.makedirs(output_path+'90perc bootstrap performance', exist_ok=True)
    subset = performances_df[(performances_df['Dataset']==dataset)&(performances_df['PredictionFunction']==predict_function)]
    fig, axis = plt.subplots(2, 3, figsize=(15,7))
    subset.hist(column=['Accuracy', 'BalancedAccuracy', 'AUROC', 'Precision', 'Recall', 'F1'], ax=axis) #bins=40
    plt.suptitle('Sampling Dist -'+' (Distance method: '+method+', Dataset: '+dataset+', Prediction function: '+predict_function+')')
    plt.tight_layout()
    plt.savefig(output_path+'90perc bootstrap performance/'+method+'_samplingdist_'+predict_function+'_'+dataset+'.png')
    plt.close()


def get_ci(metric_result):
    m = metric_result.mean()
    c = metric_result.count()
    s = metric_result.std()

    ci95_hi = m + 1.96*s 
    ci95_lo = m - 1.96*s 
    
    return round(m,4), round(ci95_lo,4), round(ci95_hi,4)


def summarize_ci(performances_df, level, method, datasets, predict_functions, metric_list, output_path):
    os.makedirs(output_path+'90perc bootstrap performance', exist_ok=True)
    ci =[]
    for predict_function in predict_functions:
        for dataset in datasets:
            all_df = performances_df[(performances_df['Dataset']==dataset)&(performances_df['PredictionFunction']==predict_function)]
            for metric in metric_list:
                metric_result = all_df[metric] 
                mean_value, ci95_low, ci95_high = get_ci(metric_result)
                ci.append([method, dataset, predict_function, metric, mean_value, ci95_low, ci95_high])
    cols =['Method', 'Dataset', 'PredictionFunction', 'Metric', 'mean', 'ci95_low', 'ci95_high']
    ci_df =pd.DataFrame(ci, columns=cols)
    ci_df.to_csv(output_path+'90perc bootstrap performance/'+method+'_'+level+'_ci.csv',index=False) 
    return ci_df


def plot_ci_all(ci_df, level, method, datasets, predict_functions, metric_list, output_path):
    os.makedirs(output_path+level+' bootstrap performance', exist_ok=True)
    for predict_function in predict_functions:
        fig, axes = plt.subplots(2, 3, figsize = (15,10)) 
        
        for ax, metric in zip(axes.ravel(), metric_list):          
            ci_subset = ci_df[(ci_df['Metric']==metric)&(ci_df['PredictionFunction']==predict_function)]
                       
            for n in range(len(datasets)):
                horizontal_line_width = 0.25
                color = '#2187bb'
                ci = ci_subset[ci_subset['Dataset']==datasets[n]]
                x = n+1
                
                left = x - horizontal_line_width / 2
                top = ci['ci95_high']
                right = x + horizontal_line_width / 2
                bottom = ci['ci95_low']
                ax.plot([x, x], [top, bottom], color=color) 
                ax.plot([left, right], [top, top], color=color)
                ax.plot([left, right], [bottom, bottom], color=color)
                ax.plot(x, ci['mean'], 'o', color='#f44336')
    
                ax.set_title(metric)
                ax.set_xticks([1, 2, 3, 4, 5, 6], datasets) 
                ax.tick_params(labelsize =6) 
                ax.set_ylim(bottom=0.3, top=1.0)
                ax.tick_params(axis='y', labelsize='medium')
                labels = ax.get_xticklabels()
                plt.setp(labels, rotation=45, horizontalalignment='right', fontsize='medium')
 
        plt.suptitle('Confidence Interval -'+' (Distance method: '+method+', Prediction function: '+predict_function+')')
        plt.tight_layout()
        plt.savefig(output_path+level+' bootstrap performance/'+method+'_ci_'+predict_function+'.png')
        plt.close()