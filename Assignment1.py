# GIRVAN NEWMAN ON WIKI-VOTE.TXT

# GIRVAN NEWMAN ON WIKI-VOTE.TXT

# GIRVAN NEWMAN ON WIKI-VOTE.TXT

import networkx as nx
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np

# Function to calculate edge betweenness centrality
def calculate_edge_betweenness(G):
    print("Calculating edge betweenness...")
    betweenness = dict.fromkeys(G.edges(), 0.0) 
    for s in G.nodes():
        S, P, sigma = single_source_shortest_path_basic(G, s)
        betweenness = accumulate_edges(betweenness, S, P, sigma, s)
    for key in betweenness:
        betweenness[key] /= 2.0 

    return betweenness

def single_source_shortest_path_basic(G, s):
    S = []  
    P = {}  
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  
    D = {}  
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]  
    while Q:  
        v = Q.pop(0)
        S.append(v)
        for w in G.neighbors(v):
            if w not in D:  
                Q.append(w)
                D[w] = D[v] + 1
            if D[w] == D[v] + 1:  
                sigma[w] += sigma[v]
                P[w].append(v)  
    return S, P, sigma

def accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        for v in P[w]:
            c = (sigma[v] / sigma[w]) * (1 + delta[w])
            if (v, w) in betweenness:
                betweenness[(v, w)] += c
            else:  
                betweenness[(w, v)] += c
            delta[v] += c
        if w != s:
            delta[w] = 0  
    return betweenness

# Girvan-Newman algorithm using the custom betweenness calculation
def girvan_newman_custom(G):
    print("Starting Girvan-Newman algorithm...")
    if len(G.nodes) == 1:
        yield [list(G.nodes())]
        return
    G = G.copy()
    iteration = 0
    components = [list(G.nodes())]
    while len(components) == 1 or G.number_of_edges() > 0:
        iteration += 1
        print(f"\nIteration {iteration}...")
        betweenness = calculate_edge_betweenness(G)
        max_edge = max(betweenness, key=betweenness.get)
        print(f"Removing edge with highest betweenness: {max_edge}, value: {betweenness[max_edge]}")
        G.remove_edge(*max_edge)
        components = list(nx.connected_components(G))
        print(f"Number of communities: {len(components)}")
        print(f"Graph edges remaining: {G.number_of_edges()}")
        yield components
        if G.number_of_edges() == 0:
            break

# Function to plot the dendrogram
def plot_dendrogram(G, levels):
    print("Creating dendrogram...")
    node_count = len(G.nodes())
    node_to_index = {node: i for i, node in enumerate(G.nodes())}

    distances = np.zeros((node_count, node_count))

    for i, level in enumerate(levels[:-1]):
        components_map = {}
        for j, component in enumerate(level):
            for node in component:
                components_map[node] = j
        
        for k in range(node_count):
            for l in range(k + 1, node_count):
                if components_map[k] != components_map[l]:
                    distances[k, l] += 1
                    distances[l, k] += 1

    Z = sch.linkage(distances, method='average')

    plt.figure(figsize=(10, 5))
    sch.dendrogram(Z, labels=list(G.nodes()), leaf_rotation=90)
    plt.show()

def load_wiki_vote_graph(filepath):
    print(f"Loading graph from {filepath}...")
    G = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    wiki_vote_file = 'data/Wiki-Vote.txt'  # Replace with the correct path to your dataset
    G = load_wiki_vote_graph(wiki_vote_file)

    levels = list(girvan_newman_custom(G))

    for i, level in enumerate(levels):
        print(f"\nFinal Result: Iteration {i + 1}: {level}")
        print(f"Graph edges remaining: {G.number_of_edges()}")

    plot_dendrogram(G, levels)



### GIRVAN NEWMAN FOR LASTFM DATASET

### GIRVAN NEWMAN FOR LASTFM DATASET

### GIRVAN NEWMAN FOR LASTFM DATASET

import networkx as nx
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd

# Function to calculate edge betweenness centrality
def calculate_edge_betweenness(G):
    betweenness = dict.fromkeys(G.edges(), 0.0)
    for s in G.nodes():
        S, P, sigma = single_source_shortest_path_basic(G, s)
        betweenness = accumulate_edges(betweenness, S, P, sigma, s)
    for key in betweenness:
        betweenness[key] /= 2.0
    return betweenness

def single_source_shortest_path_basic(G, s):
    S, P, sigma, D = [], {v: [] for v in G}, dict.fromkeys(G, 0.0), {}
    sigma[s], D[s], Q = 1.0, 0, [s]
    while Q:
        v = Q.pop(0)
        S.append(v)
        for w in G.neighbors(v):
            if w not in D:
                Q.append(w)
                D[w] = D[v] + 1
            if D[w] == D[v] + 1:
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma

def accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        for v in P[w]:
            c = (sigma[v] / sigma[w]) * (1 + delta[w])
            betweenness[(v, w) if (v, w) in betweenness else (w, v)] += c
            delta[v] += c
    return betweenness

# Girvan-Newman algorithm with improved stopping criterion
def girvan_newman_custom(G, max_iter=10):
    print("Starting Girvan-Newman algorithm with improved stopping criterion...")
    if len(G.nodes) == 1:
        yield [list(G.nodes())]
        return
    
    G = G.copy()
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    unchanged_iterations = 0
    iteration = 0

    while unchanged_iterations < max_iter and G.number_of_edges() > 0:
        iteration += 1
        print(f"\nIteration {iteration}...")
        betweenness = calculate_edge_betweenness(G)

        while num_new_components <= original_num_components:
            max_edge = max(betweenness, key=betweenness.get)
            print(f"Removing edge with highest betweenness: {max_edge}, value: {betweenness[max_edge]}")
            G.remove_edge(*max_edge)
            num_new_components = nx.number_connected_components(G)
            
            if num_new_components > original_num_components:
                break

            betweenness = calculate_edge_betweenness(G)
            unchanged_iterations += 1

            if unchanged_iterations >= max_iter:
                print(f"Stopping iteration due to reaching maximum number of unchanged iterations {max_iter}.")
                break

        components = list(nx.connected_components(G))
        print(f"Number of communities: {len(components)}")
        print(f"Graph edges remaining: {G.number_of_edges()}")
        yield components
        original_num_components = num_new_components

# Function to plot the dendrogram
def plot_dendrogram(G, levels):
    print("Creating dendrogram...")
    node_count = len(G.nodes())
    distances = np.zeros((node_count, node_count))

    for level in levels[:-1]:
        components_map = {node: idx for idx, component in enumerate(level) for node in component}
        for i in range(node_count):
            for j in range(i + 1, node_count):
                if components_map[i] != components_map[j]:
                    distances[i, j] += 1
                    distances[j, i] += 1

    Z = sch.linkage(distances, method='average')

    plt.figure(figsize=(10, 5))
    sch.dendrogram(Z, labels=list(G.nodes()), leaf_rotation=90)
    plt.show()

def load_graph_from_csv(filepath):
    print(f"Loading graph from {filepath}...")
    df = pd.read_csv(filepath)
    
    G = nx.from_pandas_edgelist(df, source='node_1', target='node_2')
    
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    csv_file = 'data/lastfm_asia_edges.csv' #Replace with the required filepath
    G = load_graph_from_csv(csv_file)

    levels = list(girvan_newman_custom(G))

    for i, level in enumerate(levels):
        print(f"\nFinal Result: Iteration {i + 1}: {level}")
        print(f"Graph edges remaining: {G.number_of_edges()}")

    plot_dendrogram(G, levels)



### LOUVAIN FOR WIKI-VOTE DATASET

### LOUVAIN FOR WIKI-VOTE DATASET

### LOUVAIN FOR WIKI-VOTE DATASET

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.cluster import hierarchy
import sys
sys.setrecursionlimit(10000)

file_path = 'data/Wiki-Vote.txt'  # Update with your file path

G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

partition = {node: node for node in G.nodes()} 
m = G.size(weight='weight')

dendrogram_data = []

# Helper Function to compute the modularity change when moving a node
def compute_modularity_change(G, i, new_community, partition, sigma_in, sigma, ki, m):
    current_community = partition[i]
    if current_community == new_community:
        return 0

    ki_in_new = sum(G[i][neighbor].get('weight', 1.0) for neighbor in G.neighbors(i) if partition[neighbor] == new_community)

    delta_sigma_in = sigma_in[new_community] + 2 * ki_in_new - sigma_in[current_community]
    delta_sigma = sigma[new_community] + ki[i] - sigma[current_community]

    delta_q = delta_sigma_in / (2 * m) - (delta_sigma / (2 * m)) ** 2 - sigma_in[new_community] / (2 * m) + (sigma[new_community] / (2 * m)) ** 2

    return delta_q


# Phase 1: Modularity Optimization with early stopping
def phase1(G, partition, m, min_change_threshold=16, max_no_change_iterations=100):
    improvement = True
    iteration = 0
    no_change_iterations = 0  
    
    while improvement:
        improvement = False
        iteration += 1
        print(f" Phase 1 - Iteration {iteration}:")
        changes = 0

        sigma_in = defaultdict(float)
        sigma = defaultdict(float)
        ki = {node: G.degree(node, weight='weight') for node in G.nodes()}

        for node in G.nodes():
            community = partition[node]
            sigma[community] += ki[node]
            for neighbor in G.neighbors(node):
                if partition[neighbor] == community:
                    sigma_in[community] += G[node][neighbor].get('weight', 1.0)

        for node in G.nodes():
            current_community = partition[node]
            best_community = current_community
            max_gain = 0

            neighbor_communities = set(partition[neighbor] for neighbor in G.neighbors(node))
            for community in neighbor_communities:
                gain = compute_modularity_change(G, node, community, partition, sigma_in, sigma, ki, m)
                if gain > max_gain:
                    max_gain = gain
                    best_community = community
            
            if best_community != current_community:
                sigma_in[current_community] -= 2 * sum(G[node][neighbor].get('weight', 1.0) for neighbor in G.neighbors(node) if partition[neighbor] == current_community)
                sigma[current_community] -= ki[node]
                
                partition[node] = best_community
                
                sigma_in[best_community] += 2 * sum(G[node][neighbor].get('weight', 1.0) for neighbor in G.neighbors(node) if partition[neighbor] == best_community)
                sigma[best_community] += ki[node]
                
                changes += 1
                improvement = True

        print(f"  Number of node changes: {changes}")
        num_communities = len(set(partition.values()))
        print(f"  Number of communities: {num_communities}")

        # Early stopping criteria
        if changes < min_change_threshold:
            no_change_iterations += 1
        else:
            no_change_iterations = 0
        
        if no_change_iterations >= max_no_change_iterations:
            print("Early stopping due to minimal changes in community structure.")
            break
        
        if changes == 0:
            break

    return partition

# Phase 2: Community Aggregation
def phase2(G, partition):
    new_G = nx.Graph()
    new_partition = {}

    for community in set(partition.values()):
        new_G.add_node(community)

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        u_new = partition[u]
        v_new = partition[v]
        
        if u_new == v_new:
            if new_G.has_edge(u_new, u_new):
                new_G[u_new][u_new]['weight'] += weight
            else:
                new_G.add_edge(u_new, u_new, weight=weight)
        else:
            if new_G.has_edge(u_new, v_new):
                new_G[u_new][v_new]['weight'] += weight
            else:
                new_G.add_edge(u_new, v_new, weight=weight)

    new_partition = {community: community for community in new_G.nodes()}

    communities = defaultdict(list)
    for node, community in partition.items():
        communities[community].append(node)
    dendrogram_data.append(list(communities.values()))

    return new_G, new_partition

# Adjusted modularity calculation
def calculate_modularity(G, partition):
    m = G.size(weight='weight')
    modularity = 0
    communities = defaultdict(list)
    
    for node, community in partition.items():
        communities[community].append(node)
    
    for community_nodes in communities.values():
        internal_edges = G.subgraph(community_nodes).size(weight='weight')
        total_degree = sum(G.degree(node, weight='weight') for node in community_nodes)
        modularity += internal_edges / m - (total_degree / (2 * m)) ** 2
    
    return modularity

# Main Louvain function (combining both phases)
def louvain_algorithm(G):
    current_partition = {node: node for node in G.nodes()}
    m = G.size(weight='weight')
    previous_modularity = calculate_modularity(G, current_partition)
    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration {iteration}: Starting Phase 1...")
        current_partition = phase1(G, current_partition, m)
        new_modularity = calculate_modularity(G, current_partition)
        num_communities = len(set(current_partition.values()))
        print(f"After Phase 1: Modularity = {new_modularity:.4f}, Number of communities = {num_communities}")

        if new_modularity - previous_modularity < 1e-4:
            print(f"Convergence achieved after {iteration} iterations with modularity: {new_modularity:.4f}")
            break
        
        print(f"Iteration {iteration}: Starting Phase 2...")
        G, current_partition = phase2(G, current_partition)
        previous_modularity = new_modularity
    
    return current_partition

# Dendrogram plotting function
def plot_louvain_dendrogram(nodes_list, dendrogram_data):
    print(f"dendrogram {dendrogram_data}")

    # Converting the dendrogram_data to a distance matrix
    n = len(nodes_list)
    dist_matrix = np.zeros((n, n))
    nodes_encoding = {node: i for i, node in enumerate(nodes_list)}
    
    for i, communities in enumerate(reversed(dendrogram_data)):
        for community in communities:
            for node1 in community:
                for node2 in community:
                    if node1 != node2:
                        dist_matrix[nodes_encoding[node1]][nodes_encoding[node2]] = i
                        dist_matrix[nodes_encoding[node2]][nodes_encoding[node1]] = i

    if np.all(dist_matrix == 0):
        print("Distance matrix contains only zeros. Cannot generate meaningful dendrogram.")
        return
    
    # Convert the distance matrix to a condensed distance matrix
    condensed_dist = hierarchy.distance.squareform(dist_matrix)

    Z = hierarchy.linkage(condensed_dist, method='ward')

    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, truncate_mode='level', p=6, show_contracted=True)
    plt.title("Louvain Community Detection Dendrogram (Wiki-Vote Dataset)")
    plt.xlabel("Clustered Data Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

final_partition = louvain_algorithm(G)
num_communities = len(set(final_partition.values()))
print(f"Final community structure has {num_communities} communities.")

plot_louvain_dendrogram(list(G.nodes()), dendrogram_data)



### LOUVAIN FOR LASTFM_ASIA_EDGES DATASET

### LOUVAIN FOR LASTFM_ASIA_EDGES DATASET

### LOUVAIN FOR LASTFM_ASIA_EDGES DATASET

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.cluster import hierarchy
import sys
sys.setrecursionlimit(10000)

file_path = 'data/lastfm_asia_edges.csv'  # Update with your file path
df = pd.read_csv(file_path)

G = nx.from_pandas_edgelist(df, source='node_1', target='node_2')

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

partition = {node: node for node in G.nodes()}  

m = G.size(weight='weight')

dendrogram_data = []

# Helper function to compute the modularity change when moving a node
def compute_modularity_change(G, i, new_community, partition, sigma_in, sigma, ki, m):
    current_community = partition[i]
    if current_community == new_community:
        return 0

    ki_in_new = sum(G[i][neighbor].get('weight', 1.0) for neighbor in G.neighbors(i) if partition[neighbor] == new_community)

    delta_sigma_in = sigma_in[new_community] + 2 * ki_in_new - sigma_in[current_community]
    delta_sigma = sigma[new_community] + ki[i] - sigma[current_community]

    delta_q = delta_sigma_in / (2 * m) - (delta_sigma / (2 * m)) ** 2 - sigma_in[new_community] / (2 * m) + (sigma[new_community] / (2 * m)) ** 2

    return delta_q

# Phase 1: Modularity Optimization
def phase1(G, partition, m):
    improvement = True
    iteration = 0
    while improvement:
        improvement = False
        iteration += 1
        print(f" Phase 1 - Iteration {iteration}:")
        changes = 0

        sigma_in = defaultdict(float)
        sigma = defaultdict(float)
        ki = {node: G.degree(node, weight='weight') for node in G.nodes()}

        for node in G.nodes():
            community = partition[node]
            sigma[community] += ki[node]
            for neighbor in G.neighbors(node):
                if partition[neighbor] == community:
                    sigma_in[community] += G[node][neighbor].get('weight', 1.0)

        for node in G.nodes():
            current_community = partition[node]
            best_community = current_community
            max_gain = 0

            neighbor_communities = set(partition[neighbor] for neighbor in G.neighbors(node))
            for community in neighbor_communities:
                gain = compute_modularity_change(G, node, community, partition, sigma_in, sigma, ki, m)
                if gain > max_gain:
                    max_gain = gain
                    best_community = community

            if best_community != current_community:
                sigma_in[current_community] -= 2 * sum(G[node][neighbor].get('weight', 1.0) for neighbor in G.neighbors(node) if partition[neighbor] == current_community)
                sigma[current_community] -= ki[node]
                
                partition[node] = best_community
                
                sigma_in[best_community] += 2 * sum(G[node][neighbor].get('weight', 1.0) for neighbor in G.neighbors(node) if partition[neighbor] == best_community)
                sigma[best_community] += ki[node]
                
                changes += 1
                improvement = True

        print(f"  Number of node changes: {changes}")
        num_communities = len(set(partition.values()))
        print(f"  Number of communities: {num_communities}")

        if changes == 0:
            break

    return partition

# Phase 2: Community Aggregation
def phase2(G, partition):
    new_G = nx.Graph()
    new_partition = {}

    for community in set(partition.values()):
        new_G.add_node(community)

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        u_new = partition[u]
        v_new = partition[v]
        
        if u_new == v_new:
            if new_G.has_edge(u_new, u_new):
                new_G[u_new][u_new]['weight'] += weight
            else:
                new_G.add_edge(u_new, u_new, weight=weight)
        else:
            if new_G.has_edge(u_new, v_new):
                new_G[u_new][v_new]['weight'] += weight
            else:
                new_G.add_edge(u_new, v_new, weight=weight)

    new_partition = {community: community for community in new_G.nodes()}

    communities = defaultdict(list)
    for node, community in partition.items():
        communities[community].append(node)
    dendrogram_data.append(list(communities.values()))

    return new_G, new_partition


# Adjusted modularity calculation
def calculate_modularity(G, partition):
    m = G.size(weight='weight')
    modularity = 0
    communities = defaultdict(list)
    
    for node, community in partition.items():
        communities[community].append(node)

    for community_nodes in communities.values():
        internal_edges = G.subgraph(community_nodes).size(weight='weight')
        total_degree = sum(G.degree(node, weight='weight') for node in community_nodes)
        modularity += internal_edges / m - (total_degree / (2 * m)) ** 2
    
    return modularity

# Main Louvain function (combining both phases)
def louvain_algorithm(G):
    current_partition = {node: node for node in G.nodes()}  
    m = G.size(weight='weight')
    previous_modularity = calculate_modularity(G, current_partition)
    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration {iteration}: Starting Phase 1...")
        current_partition = phase1(G, current_partition, m)
        new_modularity = calculate_modularity(G, current_partition)
        num_communities = len(set(current_partition.values())) 
        print(f"After Phase 1: Modularity = {new_modularity:.4f}, Number of communities = {num_communities}")

        if new_modularity - previous_modularity < 1e-4:
            print(f"Convergence achieved after {iteration} iterations with modularity: {new_modularity:.4f}")
            break
        
        print(f"Iteration {iteration}: Starting Phase 2...")
        G, current_partition = phase2(G, current_partition)  
        previous_modularity = new_modularity
    
    return current_partition


# Dendrogram plotting function
def plot_louvain_dendrogram(nodes_list, dendrogram_data):
    print(f"dendrogram {dendrogram_data}")

    n = len(nodes_list)
    dist_matrix = np.zeros((n, n))
    nodes_encoding = {node: i for i, node in enumerate(nodes_list)}
    
    for i, communities in enumerate(reversed(dendrogram_data)):
        for community in communities:
            for node1 in community:
                for node2 in community:
                    if node1 != node2:
                        dist_matrix[nodes_encoding[node1]][nodes_encoding[node2]] = i
                        dist_matrix[nodes_encoding[node2]][nodes_encoding[node1]] = i

    if np.all(dist_matrix == 0):
        print("Distance matrix contains only zeros. Cannot generate meaningful dendrogram.")
        return

    condensed_dist = hierarchy.distance.squareform(dist_matrix)

    Z = hierarchy.linkage(condensed_dist, method='ward')

    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, truncate_mode='level', p=6, show_contracted=True)

    #cutoff_height = 5  # You can choose an appropriate height based on your dendrogram
    #plt.axhline(y=cutoff_height, color='r', linestyle='--')
 
    plt.title("Louvain Community Detection Dendrogram (LastFM Dataset)")
    plt.xlabel("Clustered Data Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

final_partition = louvain_algorithm(G)
num_communities = len(set(final_partition.values()))
print(f"Final community structure has {num_communities} communities.")

plot_louvain_dendrogram(list(G.nodes()), dendrogram_data)