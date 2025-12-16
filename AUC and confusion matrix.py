#Assignment 04-Banknote authentication using UCI’s “Banknote” dataset; implement logistic regression and decision tree; evaluate via ROC/AUC and confusion matrix.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,recall_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/0555/Downloads/banknotes.csv")

X = df.drop(columns=['Class'])   
y = df['Class']                  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] 

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_value = roc_auc_score(y_test, y_proba)

print(cm)
print(f1)
print(acc)
print(precision)
print(recall)

plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()
