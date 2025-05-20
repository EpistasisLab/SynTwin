import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.linalg import eigsh
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import glob


def asc_community_creation(filepath):
    print("Running step4 asc_community")
    real_data = pd.read_csv(os.path.join(filepath,'data/synthetic.csv'))

    output_folder = os.path.join(filepath,'results/spectral_clustering')
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

    ## GRAPH NETWORK
    network = cytoscape_df.copy()
    network['Obj1'] = network['Obj1'].astype(str)
    network['Obj2'] = network['Obj2'].astype(str)

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

    node_df = pd.DataFrame({'node': node_order, 'community': labels})
    node_df.to_csv(os.path.join(output_folder, 'node_communities.csv'), index=False)

    # Plot Dendrogram
    Z = linkage(embedding, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=node_order, leaf_rotation=90)
    plt.axhline(y=distance_threshold, color='red', linestyle='--', label=f"Threshold = {distance_threshold}")
    plt.legend()
    plt.title("Dendrogram (Spectral Embedding + Agglomerative Clustering)")
    plt.xlabel("Node")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'dendrogram.png'), dpi=300)
    plt.close()

    # Save linkage matrix
    np.save(os.path.join(output_folder, 'linkage_matrix.npy'), Z)

    # Group nodes by cluster label
    community_dict = defaultdict(list)
    for node, label in zip(node_order, labels):
        community_dict[label].append(node)

    # All communities
    communities = list(community_dict.values())
    comm_non_single = [c for c in communities if len(c) > 1]
    comm_10members = [c for c in communities if len(c) >= 10]

    # Summary stats
    total_10 = sum(len(c) for c in comm_10members)

    # Label 1 rate + community sizes
    label_1_rates = []
    community_sizes = []

    for community_ids in comm_10members:
        sub_df = real_data[real_data['IID'].isin(community_ids)]
        outcome_counts = sub_df['outcome'].value_counts(normalize=True)
        label_1_rate = outcome_counts.get(1, 0)
        label_1_rates.append(label_1_rate)
        community_sizes.append(len(sub_df))

    label_1_rates = np.array(label_1_rates)
    community_sizes = np.array(community_sizes)

    # Compute weighted average label 1 rate (across ≥10 member communities)
    if len(community_sizes) > 0 and np.sum(community_sizes) > 0:
        weighted_label1_rate = np.average(label_1_rates, weights=community_sizes)
        avg_comm_size = np.mean(community_sizes)
    else:
        weighted_label1_rate = 0
        avg_comm_size = 0

    # Save label_1 rates and sizes
    comm_data_df = pd.DataFrame({
        'label_1_rate': label_1_rates,
        'community_size': community_sizes
    })
    comm_data_df.to_csv(os.path.join(output_folder, 'community_label1_rates.csv'), index=False)

    # Create result summary DataFrame
    result_df = pd.DataFrame([[
        len(comm_10members),          # num_10members_communities
        len(comm_non_single),         # num_nonsingle_communities
        len(communities),             # num_communities
        total_10,                      # total_patient_10_comm
        weighted_label1_rate,         # weighted_label1_rate
        avg_comm_size                 # avg_comm_size
    ]], columns=[
        'num_10members_communities',
        'num_nonsingle_communities',
        'num_communities',
        'total_patient_10_comm',
        'weighted_label1_rate',
        'avg_comm_size'
    ])
    result_df.to_csv(os.path.join(output_folder, 'community_summary.csv'), index=False)

    # Plot Diagnosis distribution
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    bin_indices = np.digitize(label_1_rates, bins, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)

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

    print("step4 asc_community completed")

