#Assignment 1: Simple Linear Regression on Height vs Weight Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = {
    "Height": [150, 152, 154, 156, 158, 160, 162, 164, 166, 168,
               170, 172, 174, 176, 178, 180, 182, 184, 186, 188],
    "Weight": [50, 52, 53, 54, 56, 58, 59, 61, 62, 64,
               66, 68, 69, 71, 72, 74, 76, 77, 79, 81]
}
df = pd.DataFrame(data)
print(df.head())

X = df[["Height"]]
y = df["Weight"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("  MSE :", mse)
print("  RMSE:", rmse)
print("  R²  :", r2)

plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Best Fit Line")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Linear Regression: Height vs Weight")
plt.legend()
plt.show()

new_height = pd.DataFrame([[175]], columns=["Height"])
predicted_weight = model.predict(new_height)
print(f"\nPredicted weight for 175 cm: {predicted_weight[0]:.2f} kg")