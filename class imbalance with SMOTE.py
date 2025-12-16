# Assignment 5: Wine dataset; implement at least two algorithms, handle class imbalance with SMOTE or weighting, compare metrics.  

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\0555\Downloads\wine-class.csv")

X = df.drop(columns=['class'])
y = df['class']

print("Input samples:\n", X.head())
print("Output samples:\n", y.head())

print("\nOriginal Class Distribution:\n", df['class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

print("\nScaled Training Samples:\n", pd.DataFrame(X_train).head())

X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

print("\nBalanced Class Distribution After SMOTE:\n",
      pd.Series(y_train_res).value_counts().sort_index())

lr = LogisticRegression(max_iter=5000, multi_class='multinomial')
lr.fit(X_train_res, y_train_res)

print("\n--- Logistic Regression Results ---")
y_pred_lr = lr.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_res, y_train_res)

print("\n--- Random Forest Results ---")
y_pred_rf = rf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

print("\nModel Comparison:")
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

if acc_lr > acc_rf:
    print(f"Final Choice: Logistic Regression (Accuracy = {round(acc_lr*100,2)}%)")
else:
    print(f"Final Choice: Random Forest (Accuracy = {round(acc_rf*100,2)}%)")