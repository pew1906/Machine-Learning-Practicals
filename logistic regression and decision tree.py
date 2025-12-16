#Assignment 3: Student grade prediction using logistic regression and decision tree; compare accuracies and visualize.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/0555/Downloads/grades.csv")

X = df[['hours_studied', 'attendance_pct']]
y = df['grade_A']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log = LogisticRegression().fit(X_train, y_train)
tree = DecisionTreeClassifier().fit(X_train, y_train)

acc_log = accuracy_score(y_test, log.predict(X_test))
acc_tree = accuracy_score(y_test, tree.predict(X_test))

print("Logistic Regression Accuracy:", acc_log)
print("Decision Tree Accuracy:", acc_tree)

print("\nBest Model:", "Logistic Regression" if acc_log > acc_tree else "Decision Tree")

models=['logistic regression','decision tree']
accuracies=[acc_log,acc_tree]

plt.bar(models,accuracies)
plt.show()