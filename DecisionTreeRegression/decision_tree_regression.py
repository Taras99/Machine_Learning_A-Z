# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Removed leading slash
X = dataset.iloc[:, 1:2].values  # Position level
y = dataset.iloc[:, 2].values    # Salary

# Note: Decision Trees don't require feature scaling
# Note: For this small dataset, we're using all data for training

# Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting on the training data
y_pred = regressor.predict(X)

# Calculating evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Printing evaluation metrics
print("\nDecision Tree Regression Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# Predicting a new result
position_level = 6.5
predicted_salary = regressor.predict([[position_level]])[0]  # Fixed prediction format
print(f"\nPredicted salary for position level {position_level}: ${predicted_salary:,.2f}")

# Visualising the Decision Tree Regression results (higher resolution)
plt.figure(figsize=(10, 6))
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Salary vs Position Level (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary ($)')
plt.legend()
plt.show()