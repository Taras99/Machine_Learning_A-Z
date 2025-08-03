# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importing the dataset
dataset = pd.read_csv('/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  # Using Annual Income (column 3) and Spending Score (column 4)


# Using the dendrogram to find the optimal number of clusters
plt.figure(figsize=(12, 8))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram for Hierarchical Clustering', fontsize=14)
plt.xlabel('Customers', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.axhline(y=150, color='r', linestyle='--')  # Visual line to help determine cluster count
plt.grid(axis='y', alpha=0.5)
plt.show()

# Fitting Hierarchical Clustering to the dataset
hc = AgglomerativeClustering(
    n_clusters=5, 
    metric='euclidean',  # Changed from 'affinity' to 'metric'
    linkage='ward'
)
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
cluster_labels = ['Careful Spenders', 'Standard Customers', 'Target Customers', 
                 'Careless Spenders', 'Sensible Spenders']

for cluster in range(5):
    plt.scatter(
        X[y_hc == cluster, 0], 
        X[y_hc == cluster, 1], 
        s=100, 
        c=colors[cluster], 
        label=cluster_labels[cluster]
    )

plt.title('Customer Segmentation using Hierarchical Clustering', fontsize=14)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(fontsize=10, title='Customer Groups')
plt.grid(alpha=0.3)
plt.show()