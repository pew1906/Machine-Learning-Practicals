#Assignment 10: Perform Hierarchical Clustering on Wholesale Customer Data

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\0555\ML assisgnments\Wholesale customers data.csv")

data.head()
numerical_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = data[numerical_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(Z, labels=data.index)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
data['Cluster'] = clusters

# Step 8: Analyze clusters
cluster_summary = data.groupby('Cluster')[numerical_features].mean().round(2)
cluster_size = data['Cluster'].value_counts().sort_index()
cluster_summary['Size'] = cluster_size

print("\nCluster Summary:")
print(cluster_summary)

# Step 9: Optional – inspect cluster members
print("\nSample of cluster assignments:")
print(data[['Cluster'] + numerical_features].head(10))