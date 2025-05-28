import pandas as pd
import glob
import os
import sys
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from ast import literal_eval
from cdist_gower import cdist_gower
from datetime import datetime
import pylab as pl
import math    

from utils import compute_performance, predict_outcome, summarize_ci, plot_ci_all
from utils import bootstrap_resample_90perc_performance, show_metric_dist


def outcome_prediction_asc(filepath, threshold=0.5):
    print("Running step5 outcome prediction with asc community")
    print(datetime.now())

    filepath = filepath +'/'
    data_path = filepath + 'data/'
    output_path = filepath + 'results/vital_prediction_asc/' 
    distance_folder ='results/'
    percolation_folder = 'results/percolation_threshold/'
    method = 'gower'

    os.makedirs(output_path, exist_ok=True)

    # Redirect stdout to a file
    log_file = os.path.join(output_path, 'log.txt')
    sys.stdout = open(log_file, 'w')

    random_state = 46 # Set the best random state calculated from the previous step

    id_col = 'IID'
    predict_col = 'outcome'

    real_le = pd.read_csv(filepath+'syntwin_gs/gene_scores_test_alzkb.csv') 
    real_data = real_le.copy()
    real_le = real_le.drop(columns= predict_col) 

    synth_filename = 'synth_GC.csv'  # synth_CTGAN or synth_GC or synth_TVAE
    synth_le = pd.read_csv(data_path+synth_filename)  
    synthetic_data = synth_le.copy() # Modify this line if the column name is different
    synth_ids = synthetic_data[id_col]
    synth_le = synth_le.drop(columns = [id_col, predict_col]) # Modify this line if the column name is different      

    datasets  = ['real','synth','synth_topn','real+synth','real+synth_topn','real_OC']
    predict_functions = ['mean','mode', 'knn'] 
    metric_list = ['Accuracy', 'BalancedAccuracy', 'AUROC', 'Precision', 'Recall', 'F1']

    # Keep num features only
    features = real_le.drop(columns=[id_col])
    num_cols = features.shape[1]
    cat_features = [False] * num_cols
    num_max = np.ones(num_cols)
    num_ranges = np.zeros(num_cols)
    for idx, col_name in enumerate(features.columns):
        col_array = features[col_name].astype(np.float32).values
        max_val = np.nanmax(col_array)
        min_val = np.nanmin(col_array)

        if np.isnan(max_val):
            max_val = 0.0
        if np.isnan(min_val):
            min_val = 0.0

        num_max[idx] = max_val
        num_ranges[idx] = np.abs(1 - min_val / max_val) if max_val != 0 else 0.0        

    random.seed(random_state)
    network = pd.read_csv(filepath+ percolation_folder+'cytoscape_'+method+'_'+str(threshold)+'.csv') 
    network.loc[:,'Obj1'] = network.loc[:,'Obj1'].astype(str)
    network.loc[:,'Obj2'] = network.loc[:,'Obj2'].astype(str)

    # Compute similarity weights from distance
    network['Weights'] = 1 - network['Dist'] / network['Dist'].max()

    # Build a weighted graph
    G = nx.from_pandas_edgelist(
        network,
        source='Obj1',
        target='Obj2',
        edge_attr='Weights'
    )

    # Compute Spectral Embedding
    node_order = sorted(G.nodes())  # fix order
    L = nx.normalized_laplacian_matrix(G, nodelist=node_order).astype(float)
    n_components = 20  # spectral dimension

    # Compute top-k eigenvectors
    vals, vecs = eigsh(L, k=n_components+1, which='SM')
    embedding = normalize(vecs[:, 1:])

    # Agglomerative Clustering (by distance threshold)
    distance_threshold = 3.0 #Lower: more smaller clusters; Higher: fewer larger clusters
    model = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        linkage='ward'
    )
    labels = model.fit_predict(embedding)

    # Group nodes by cluster label
    community_dict = defaultdict(list)
    for node, label in zip(node_order, labels):
        community_dict[label].append(node)

    # All communities
    communities = list(community_dict.values())    
    comm_non_single = [c for c in communities if len(c) > 1]
    comm_10members = [c for c in communities if len(c) >= 10]   
    
    #stats of number of communities (non-single, 10-memembers)
    num_non_single_comm=len(comm_non_single)
    num_10members_comm=len(comm_10members)
    num_nodes=sum([len(c) for c in comm_10members])

    print(str(len(communities))+' communities')
    print(str(num_non_single_comm)+' non single communities')
    print(str(num_10members_comm)+' 10+ members communities')
    print(str(num_nodes)+' nodes in 10+ members communities')

    #distance
    input_file = filepath+distance_folder+'/'+method+'_real_real.pkl'
    real_real = pd.read_pickle(input_file)
    
    # community info
    community_centers = []
    community_member_lists = []
    community_distances = []
    community_info =[]

    n_communities_10 = 0
    #for each community with >=10 memebers, get the center, list of members, distance from each member to center, num of dead patients
    for members in communities:
        if len(members) >= 10:
            random.seed(random_state)
            subgraph = G.subgraph(members)
            
            # check connected-component
            if not nx.is_connected(subgraph):
                components = list(nx.connected_components(subgraph))
                largest = max(components, key=len)
                subgraph = subgraph.subgraph(largest).copy()
            
            centrality = nx.eigenvector_centrality_numpy(subgraph, weight='Weights') 
            community_center = max(centrality, key=centrality.get)
            community_centers.append(community_center)

            # Member list (converted if needed)
            community_members = [literal_eval(mem) if isinstance(mem, str) and mem.startswith('(') else mem for mem in members]
            community_member_lists.append(community_members)

            # Community distance
            dist_real = real_real.loc[community_members, community_center].to_list()
            community_distance = max(dist_real)
            community_distances.append(community_distance)

            outcome_real = real_data.loc[real_data['IID'].isin(community_members), 'outcome']

            outcome1 = outcome_real.sum() 

            outcome1_ratio = outcome1 / len(community_members)

            n_communities_10+=1

            community_info.append([n_communities_10, len(community_members), outcome1, outcome1_ratio, community_distance, community_center])       
                
    cols = ['n_communities_10', 'num_patients', 'num_outcome1', 'outcome1_ratio', 'community_distance', 'community_center']
    community_info_df = pd.DataFrame(community_info, columns=cols)
    community_info_df.to_csv(output_path+method+'_community_info.csv', index=False)

    del real_real

    # prediction
    predic_function_a =[]
    predic_function_b =[]
    predic_function_bn =[]
    predic_function_c =[]   
    predic_function_cn =[]
    predic_function_d =[]

    for i in range(n_communities_10):
        
        community_id = i+1
        
        # calculate the distance between center and synthetic patients 
        center = real_le[real_le['IID']==community_centers[i]].drop(columns=['IID'])  
        random.seed(random_state)
        synth_temp = pd.DataFrame(cdist_gower(center, synth_le.iloc[:,0:], cat_features = cat_features, num_max=num_max, num_ranges=num_ranges), columns=synth_ids)

        # filter synthetic patients to those within the community (distance to center less than max distance )
        synth_community_members = synth_temp.columns[synth_temp.iloc[0].lt(community_distances[i])].tolist()


        # for each real patient in the community, predict vital status with the six prediction functions (the names of prediction functions are slightly different from paper)
        for j in range(len(community_member_lists[i])):        
            real_node = community_member_lists[i][j]

            #a (real patients)
            comm_id = community_member_lists[i]
            a_id = [x for x in comm_id if x != real_node] # get rid of itself in the list
            real_label = real_data.loc[real_data['IID'].isin(a_id), 'outcome']
            real_feature = real_data[real_data['IID'].isin(a_id)].drop(columns={'IID', 'outcome'}).values

            count_a, std_a, dead_a, ratio_a, pred_label_mean_a, pred_label_mode_a, pred_label_knn_a, true_label = predict_outcome(real_data, real_label, real_feature, real_node, id_col, predict_col)        
            predic_function_a.append([community_id, real_node, method, 'real', count_a, std_a, dead_a, ratio_a, pred_label_mean_a, pred_label_mode_a, pred_label_knn_a, true_label])

            if len(synth_community_members)>0:
                #b  (digital twins)
                b_id = synth_community_members
                synth_label = synthetic_data.loc[synthetic_data['IID'].isin(b_id), 'outcome']
                synth_feature = synthetic_data[synthetic_data['IID'].isin(b_id)].drop(columns={'IID', 'outcome'}).values

                count_b, std_b, dead_b, ratio_b, pred_label_mean_b, pred_label_mode_b, pred_label_knn_b, true_label = predict_outcome(real_data, synth_label, synth_feature, real_node, id_col, predict_col)                
                predic_function_b.append([community_id, real_node, method, 'synth', count_b, std_b, dead_b, ratio_b, pred_label_mean_b, pred_label_mode_b, pred_label_knn_b, true_label])

                #b top n (the closest digital twins )
                real_node_le = real_le[real_le['IID']==real_node].drop(columns=['IID'])  
                synth_community_members_le = synthetic_data[synthetic_data['IID'].isin(synth_community_members)].drop(columns={'IID', 'outcome'})
                random.seed(random_state)
                synth_to_real_node = pd.DataFrame(cdist_gower(real_node_le, synth_community_members_le, cat_features = cat_features, num_max=num_max, num_ranges=num_ranges), columns=synth_community_members)
                
                btopn_id = synth_to_real_node.loc[0].nsmallest(n=len(a_id), keep='all').index.tolist()
                synthn_label = synthetic_data.loc[synthetic_data['IID'].isin(btopn_id), 'outcome']
                synthn_feature = synthetic_data[synthetic_data['IID'].isin(btopn_id)].drop(columns={'IID', 'outcome'}).values

                count_bn, std_bn, dead_bn, ratio_bn, pred_label_mean_bn, pred_label_mode_bn, pred_label_knn_bn, true_label = predict_outcome(real_data, synthn_label, synthn_feature, real_node, id_col, predict_col)                
                predic_function_bn.append([community_id, real_node, method, 'synth_topn', count_bn, std_bn, dead_bn, ratio_bn, pred_label_mean_bn, pred_label_mode_bn, pred_label_knn_bn, true_label])
                
                #c (real patients and digital twins)
                all_label = pd.concat([real_label, synth_label]) #np.concatenate((real_label.values, synth_label.values), axis=0)
                all_feature = np.concatenate((real_feature, synth_feature), axis=0)

                count_c, std_c, dead_c, ratio_c, pred_label_mean_c, pred_label_mode_c, pred_label_knn_c, true_label = predict_outcome(real_data, all_label, all_feature, real_node, id_col, predict_col)                
                predic_function_c.append([community_id, real_node, method, 'real+synth', count_c, std_c, dead_c, ratio_c, pred_label_mean_c, pred_label_mode_c, pred_label_knn_c, true_label])

                #c top n (real patients and closest digital twins)
                alln_label = pd.concat([real_label, synthn_label]) 
                alln_feature = np.concatenate((real_feature, synthn_feature), axis=0)

                count_cn, std_cn, dead_cn, ratio_cn, pred_label_mean_cn, pred_label_mode_cn, pred_label_knn_cn, true_label = predict_outcome(real_data, alln_label, alln_feature, real_node, id_col, predict_col)                
                predic_function_cn.append([community_id, real_node, method, 'real+synth_topn', count_cn, std_cn, dead_cn, ratio_cn, pred_label_mean_cn, pred_label_mode_cn, pred_label_knn_cn, true_label])

            else:
                predic_function_b.append([community_id, real_node, method, 'synth', 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, true_label])
                predic_function_bn.append([community_id, real_node, method, 'synth_topn', 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, true_label])
                predic_function_c.append([community_id, real_node, method, 'real+synth', count_a, std_a, dead_a, ratio_a, pred_label_mean_a, pred_label_mode_a, pred_label_knn_a, true_label])
                predic_function_cn.append([community_id, real_node, method, 'real+synth_topn', count_a, std_a, dead_a, ratio_a, pred_label_mean_a, pred_label_mode_a, pred_label_knn_a, true_label])

            #d (real patients outside the community)
            real_id = real_data['IID']
            res_id = [i for i in real_id if i not in comm_id]
            random.seed(random_state)
            d_id = random.sample(res_id, len(community_member_lists[i])-1) 
            outer_label = real_data.loc[real_data['IID'].isin(d_id), 'outcome']
            outer_feature = real_data[real_data['IID'].isin(d_id)].drop(columns={'IID', 'outcome'}).values

            count_d, std_d, dead_d, ratio_d, pred_label_mean_d, pred_label_mode_d, pred_label_knn_d, true_label = predict_outcome(real_data, outer_label, outer_feature, real_node, id_col, predict_col)                
            predic_function_d.append([community_id, real_node, method, 'real_OC', count_d, std_d, dead_d, ratio_d, pred_label_mean_d, pred_label_mode_d, pred_label_knn_d, true_label])

    #cols =['CommunityId', 'Method', 'Dataset', 'PredictionFunction', 'Precision', 'Recall', 'Accuracy', 'BalancedAccuracy','F1', 'AUROC', 'CM']
    #community_performances_df = pd.DataFrame(performances, columns=cols)
    #community_performances_df.to_csv(output_path+method+'_community_performances.csv', index=False)

    cols =['CommunityId', 'Real_Node', 'Method', 'Dataset', 'count', 'std', 'dead', 'ratio', 'mean', 'mode', 'knn', 'Real_VitalStatus']
    predic_function_a_df= pd.DataFrame(predic_function_a, columns=cols)
    predic_function_b_df= pd.DataFrame(predic_function_b, columns=cols)
    predic_function_bn_df= pd.DataFrame(predic_function_bn, columns=cols)
    predic_function_c_df= pd.DataFrame(predic_function_c, columns=cols)
    predic_function_cn_df= pd.DataFrame(predic_function_cn, columns=cols)
    predic_function_d_df= pd.DataFrame(predic_function_d, columns=cols)

    all_predict= pd.concat([predic_function_a_df, predic_function_b_df, predic_function_bn_df, predic_function_c_df, predic_function_cn_df, predic_function_d_df])
    #all_predict= all_predict.set_index(['Real_Node','Method','Dataset'])
    all_predict.to_csv(output_path+method+'_all_predict.csv')


    # Calculate overall performance
    overall_performances= []
    for dataset in datasets:
        all_df = all_predict[(all_predict['Dataset']==dataset)]
        for predict_function in predict_functions:
            cols_to_select = [col for col in all_predict.columns if predict_function in col]
            cols_to_select.append('Real_VitalStatus')
            df = all_df[cols_to_select]
            df = df.dropna(subset=[predict_function])

            y_true = df['Real_VitalStatus']
            y_pred = df[predict_function]

            overall_performance_list = compute_performance(y_true=y_true, y_predict=y_pred)
            overall_performance = [method, dataset, predict_function]
            overall_performance.extend(overall_performance_list)
            overall_performances.append(overall_performance) 

    cols = ['Method', 'Dataset', 'PredictionFunction', 'Precision', 'Recall', 'Accuracy', 'BalancedAccuracy','F1', 'AUROC', 'CM']
    overall_performances_df= pd.DataFrame(overall_performances, columns=cols)
    overall_performances_df.to_csv(output_path+'overall_performances.csv',index=False)

    # Calculate overall performance for 90% bootstrap
    performances_df = bootstrap_resample_90perc_performance(all_predict, method, datasets, predict_functions, output_path)
    ci_df = summarize_ci(performances_df, '90perc', method, datasets, predict_functions, metric_list, output_path)
    plot_ci_all(ci_df, '90perc', method, datasets, predict_functions, metric_list, output_path)

    # Show sample distribution of performance metrics
    for dataset in datasets:
        for predict_function in predict_functions:
            show_metric_dist(performances_df, method, predict_function, dataset, output_path)

    print(datetime.now())

    sys.stdout.close()
    sys.stdout = sys.__stdout__

    print("step5 outcome prediction with asc community completed")