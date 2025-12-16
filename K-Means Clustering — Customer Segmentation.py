#Assignment 6: K-Means Clustering — Customer Segmentation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv(r"C:\Users\0555\Downloads\Mall_Customers.csv")

X = data.iloc[:, [3, 4]]

sse = []
for k in range(1, 16):
    cls = KMeans(n_clusters=k, random_state=0)
    cls.fit(X)
    sse.append(cls.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method")
plt.show()

silh = []
for k in range(2, 16):
    cls = KMeans(n_clusters=k, random_state=0)
    labels = cls.fit_predict(X)
    score = silhouette_score(X, labels)
    silh.append(score)

plt.figure(figsize=(8, 5))
plt.bar(range(2, 16), silh, color="skyblue")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

best_k = silh.index(max(silh)) + 2
print("Best K (from Silhouette):", best_k)

cls = KMeans(n_clusters=best_k, random_state=0)
labels = cls.fit_predict(X)

data["Cluster"] = labels

plt.figure(figsize=(10, 7))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="rainbow", alpha=0.7)
centers = cls.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=300, c="red", marker="X", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"KMeans Clustering (k={best_k})")
plt.legend()
plt.show()

new_customer = [[30, 73]]  
pred_cluster = cls.predict(new_customer)[0]
print("New customer belongs to Cluster:", pred_cluster)

if 4 in set(labels):
    data[data["Cluster"] == 4].to_csv("Mall_Customers_Four.csv", index=False)
