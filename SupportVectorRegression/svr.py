# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importing the dataset
dataset = pd.read_csv('/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Position level
y = dataset.iloc[:, 2].values   # Salary

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

# Fitting SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled)

# Predicting on the entire dataset (since we're not splitting)
y_pred_scaled = regressor.predict(X_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Calculating evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Printing evaluation metrics
print("\nRegression Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# Predicting a new result
position_level = 6.5
scaled_position = sc_X.transform(np.array([[position_level]]))
scaled_prediction = regressor.predict(scaled_position)
y_pred_new = sc_y.inverse_transform(scaled_prediction.reshape(-1, 1))
print(f"\nPredicted salary for position level {position_level}: ${y_pred_new[0][0]:,.2f}")

# Visualising the SVR results (original scale)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual Salary')
X_plot = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
X_plot_scaled = sc_X.transform(X_plot)
plt.plot(X_plot, sc_y.inverse_transform(regressor.predict(X_plot_scaled).reshape(-1, 1)), 
         color='blue', label='SVR Prediction')
plt.title('Salary vs Position Level (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary ($)')
plt.legend()
plt.show()