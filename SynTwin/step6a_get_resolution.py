
method = 'gower'
#['euclidean','manhattan','cosine','gower']

# path_to_data is the folder for all input and output of SynTwin, replace path_to_data to your path
# Distance_matrices and Percolation_threshold subfolders were created in previous step
path = 'path_to_data/'
distance_folder ='Distance_matrices'
percolation_folder = 'Percolation_threshold'
output_path = path

def get_threshold(method):
    if method == 'euclidean':
        threshold=2
    if method == 'manhattan':
        threshold=4
    if method == 'cosine':
        threshold=0.22 
    if method == 'gower':
        threshold=0.12  
    return threshold

#the resolution parameter to use in the modularity measure. Smaller values result in a smaller number of larger clusters, while higher values yield a large number of small clusters. The classical modularity measure assumes a resolution parameter of 1.
import pandas as pd
import igraph as ig
from igraph import Graph
import numpy as np
import random


print(method)
threshold=get_threshold(method) 

# detect communities with different resolution parameter
# repeat 100 time with random seed from 1 to 100
df=pd.DataFrame()       
for m in range(100):
    random.seed(m)
    print('random_state: '+str(m))

    #To prevent an uninformative fully connected network, we used a percolation threshold equal to the first upward inflection point of the convex part of the sigmoid relationship between edge weight (X axis) and network size (Y axis) as an objective approach to filtering edges.
    network = pd.read_csv(path+percolation_folder+'/cytoscape_'+method+str(threshold)+'.csv') 
    network.loc[:,'Obj1'] = network.loc[:,'Obj1'].astype(str)
    network.loc[:,'Obj2'] = network.loc[:,'Obj2'].astype(str)

    #build a network with 
    g = Graph.DataFrame(network, directed=False, use_vids=False)
    dist = network['Dist'].tolist()
    max_dist = np.max(dist)
    weights = [ 1 - x / max_dist for x in dist]
    g.es['weight'] = weights


    #detect communities
    result =[]
    for n in range(100,1000,10): #100,1000,10 for all + 900,1400,10 for gower
        random.seed(m)
        communities = g.community_multilevel(weights=g.es['weight'], resolution=n)

        comm_non_single = []
        comm_10members = []
        for i in range(len(communities)):
            if len(communities[i])>1:
                comm_non_single.append(communities[i])            
            if len(communities[i])>=10:
                comm_10members.append(communities[i])
        result.append([method, m, n, len(comm_10members),len(comm_non_single),len(communities)])

    #save numbers of 10 members communities
    result_df= pd.DataFrame(result, columns=(['method','random_state','resolution','num_10members_communities','num_nonsingle_communities', 'num_communities']))
    df = pd.concat([df,result_df])
    #we split the run to two part for gower, the results are saved in resolution1000_all_'+method+'.csv & resolution1000_all_'+method+'1.csv 
    df.to_csv(output_path+percolation_folder+'/resolution1000_all_'+method+'.csv')

#for each distance method and random seed, find the resolution with the max 10members_communities
best_df = df.sort_values(by='num_10members_communities', ascending=False).groupby(['method','random_state']).head(1)
best_df.to_csv(output_path+percolation_folder+'/resolution1000_count_'+method+'.csv')

#get the most frequent resolution over the 100 replicates for each distance method
best_count = best_df.groupby('method')['resolution'].value_counts().sort_values(ascending=False).head(5)
best_count.to_csv(output_path+percolation_folder+'/resolution1000_'+method+'.csv')
print(best_count)

