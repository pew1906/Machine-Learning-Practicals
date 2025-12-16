#Assignment 2: Multiple regression on house prices dataset; evaluate and visualize results.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\0555\Downloads\house_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade", "sqft_above"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print(features, model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance on Test Set:")
print("  MSE :", mse)
print("  RMSE:", rmse)
print("  R²  :", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Multiple Regression: Actual vs Predicted House Prices")
plt.show()

