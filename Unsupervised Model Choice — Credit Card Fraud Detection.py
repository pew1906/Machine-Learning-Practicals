#Assignment 8: Unsupervised Model Choice — Credit Card Fraud Detection
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\0555\creditcard.csv")

feature_cols = ['Time', 'Amount'] + [f"V{i}" for i in range(1,29)]
X = df[feature_cols].values
y = df['Class'].values
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 2
print("Using KMeans with k =", k)
km = KMeans(n_clusters=k, random_state=42)
assigned = km.fit_predict(X_scaled)  
centers = km.cluster_centers_
contamination = max(1e-6, y.mean())
distances=np.linalg.norm(X_scaled - centers[assigned], axis=1)
threshold = np.quantile(distances, 1.0 - contamination)

y_pred = (distances > threshold).astype(int)
print(classification_report(y, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y, y_pred))

roc_auc = roc_auc_score(y, distances)   
pr_auc  = average_precision_score(y, distances)
print(f"ROC AUC (distance score): {roc_auc:.4f}")
print(f"PR  AUC (distance score): {pr_auc:.4f}")

plt.figure(figsize=(8,4))
plt.scatter(range(len(distances)), distances, c=y_pred, cmap='coolwarm', s=5)
plt.xlabel("Transaction index")
plt.ylabel("Distance to cluster center")
plt.title("Distance of transactions (red = predicted anomaly)")
plt.show()