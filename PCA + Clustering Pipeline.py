#Assignment 7: PCA + Clustering Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.read_csv(r"C:\Users\0555\Downloads\movies .csv")
print(data.head())

X = data[["Budget", "Celebrities"]]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
cls = KMeans(n_clusters=3, random_state=0)

labels = cls.fit_predict(X_pca)
score = silhouette_score(X_pca, labels)
print(f"Silhouette Score: {score:.4f}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="rainbow")
plt.show()