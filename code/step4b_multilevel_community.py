import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import igraph as ig
import os
import glob


def multilevel_community_creation(filepath):
    print("Running step4 multilevel_community")

    real_data = pd.read_csv(os.path.join(filepath,'GS_gwas/gene_scores_test_gwas.csv'))

    output_folder = os.path.join(filepath,'results/multilevel_clustering')
    os.makedirs(output_folder, exist_ok=True)

    # threshold_to_cytoscape
    folder = os.path.join(filepath, 'results/percolation_threshold')
    pattern = os.path.join(folder, 'cytoscape_gower_*.csv')

    # Search for the single matching file
    files = glob.glob(pattern)

    # Expect exactly one file
    if len(files) == 1:
        cytoscape_df = pd.read_csv(files[0])
        print(f"Loaded file: {files[0]}")
    elif len(files) == 0:
        raise FileNotFoundError("No file matching 'cytoscape_gower_*.csv' found.")
    else:
        raise ValueError("Multiple files matching 'cytoscape_gower_*.csv' found. Please refine the pattern.")

    # get resolution df
    df_re = pd.DataFrame()

    for m in range(10):
       random.seed(m)

       # Copy input DataFrame
       network = cytoscape_df.copy()
       network['Obj1'] = network['Obj1'].astype(str)
       network['Obj2'] = network['Obj2'].astype(str)

       # Calculate weights from 'Dist'
       max_dist = network['Dist'].max()
       network['Weights'] = 1 - network['Dist'] / max_dist

       # Create iGraph graph
       g = ig.Graph.DataFrame(network, directed=False, use_vids=False)
       g.es['weight'] = network['Weights'].tolist()

       result = []
       for n in range(0, 10, 1):
           random.seed(m)
           communities = g.community_multilevel(weights=g.es['weight'], resolution=n)

           comm_non_single = [c for c in communities if len(c) > 1]
           comm_10members = [c for c in communities if len(c) >= 10]

           total_10 = sum(len(c) for c in comm_10members)

           result.append([m, n, len(comm_10members), len(comm_non_single), len(communities), total_10])

           print(f"resolution {n} completed")

       result_df = pd.DataFrame(result, columns=[
           'random_state', 'resolution', 'num_10members_communities',
           'num_nonsingle_communities', 'num_communities', 'total_patient_10_comm'
       ])
       
       df_re = pd.concat([df_re, result_df])

       print(f"random seed {m} completed")

    # Save all results
    df_re.to_csv(os.path.join(output_folder,"gene_score_resolution_all.csv"), index=False)

    # Get best resolution per random seed (max number of 10+ member communities)
    best_df = df_re.sort_values(by='num_10members_communities', ascending=False).groupby('random_state').head(1)
    best_df = best_df.sort_values(by='random_state')
    best_df.to_csv(os.path.join(output_folder, "gene_score_resolution_df.csv"), index=False)

    best_count = best_df['resolution'].value_counts().sort_values(ascending=False).head(5)
    best_count.to_csv(os.path.join(output_folder,'gene_score_resolution_best.csv'))

    # Get the most frequent resolution value
    most_frequent_resolution = best_count.index[0]

    # previous calling .loc[] using an index that’s duplicated (e.g., 7), resulting in multiple rows returned
    filtered = best_df[best_df['resolution'] == most_frequent_resolution]
    row_with_max_value = filtered.sort_values(by='total_patient_10_comm', ascending=False).iloc[0]
    print(row_with_max_value)

    # Create graph network with the best resolution and random state
    best_random_state = int(row_with_max_value['random_state'])
    print(best_random_state)
    best_resolution = row_with_max_value['resolution']
    print(best_resolution)

    network = cytoscape_df.copy()
    network.loc[:,'Obj1'] = network.loc[:,'Obj1'].astype(str)
    network.loc[:,'Obj2'] = network.loc[:,'Obj2'].astype(str)

    # apply multi-level algo with weight
    network_weights = network.copy()
    network_weights['Weights'] = 1 - network_weights['Dist']/max(network_weights['Dist'])
    random.seed(best_random_state)
    graph_weighted = ig.Graph.DataFrame(network_weights, directed=False, use_vids=False)
    communities = graph_weighted.community_multilevel(weights='Weights', resolution=best_resolution)

    result = []
    comm_non_single = [c for c in communities if len(c) > 1]
    comm_10members = [c for c in communities if len(c) >= 10]
    total_10 = sum(len(c) for c in comm_10members)
    result.append([best_resolution, best_random_state, len(comm_10members), len(comm_non_single), len(communities), total_10])

    result_df = pd.DataFrame(result, columns=[
        'resolution', 'random_state', 'num_10members_communities', 'num_nonsingle_communities', 'num_communities', 'total_patient_10_comm'])
    result_df.to_csv(os.path.join(output_folder, "gene_score_resolution_best_communities.csv"), index=False)

    # find "diagnosis" in each community
    graph_weighted = ig.Graph.DataFrame(network_weights, directed=False, use_vids=False)

    # Map node index to ID
    index_to_id = graph_weighted.vs['name']

    # Convert communities from index to ID
    comm_10members_ids = [[index_to_id[i] for i in comm] for comm in comm_10members]

    # Calculate label 1 rate and community size
    label_1_rates = []
    community_sizes = []
    outcome_dist_rows = []

    for idx, community_ids in enumerate(comm_10members_ids):
        sub_df = real_data[real_data['IID'].isin(community_ids)]
        outcome_counts = sub_df['outcome'].value_counts(normalize=True)
        label_1_rate = outcome_counts.get(1, 0)  # default to 0 if no label 1
        label_1_rates.append(label_1_rate)
        community_sizes.append(len(sub_df))
        
        # Save outcome distribution
        row = {'Community': f"Community_{idx+1}", 'Size': len(sub_df)}
        for label, val in outcome_counts.items():
            row[f"Label_{label}_rate"] = val
        outcome_dist_rows.append(row)

    # Save community label 1 rates and sizes
    comm_df = pd.DataFrame({
        'Community': [f"Community_{i+1}" for i in range(len(comm_10members_ids))],
        'Community_Size': community_sizes,
        'Label_1_Rate': label_1_rates
    })
    comm_df.to_csv(os.path.join(output_folder, 'community_stats.csv'), index=False)

    # Save outcome distribution per community
    outcome_dist_df = pd.DataFrame(outcome_dist_rows)
    outcome_dist_df.to_csv(os.path.join(output_folder, 'outcome_distributions.csv'), index=False)

    # Convert lists to numpy arrays
    label_1_rates = np.array(label_1_rates)
    community_sizes = np.array(community_sizes)

    # Define bins for label 1 rate (0–1 in 0.1 steps)
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    # Assign each community to a bin
    bin_indices = np.digitize(label_1_rates, bins, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels)-1)

    # Sum community sizes in each bin
    volume_per_bin = np.zeros(len(bin_labels))
    for idx, size in zip(bin_indices, community_sizes):
        volume_per_bin[idx] += size

    plt.figure(figsize=(8, 4))
    plt.bar(bin_labels, volume_per_bin)
    plt.xlabel("Diagnosis Rate in Community (Proportion of Label 1)")
    plt.ylabel("Total Community Count")
    plt.title("Community Diagnosis Rate Distribution (≥10 Members)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'diagnosis_rate_distribution.png'), dpi=300)
    plt.close()

    print("step4 multilevel_community completed")

    return best_resolution, best_random_state
