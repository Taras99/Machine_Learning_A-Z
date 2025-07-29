# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Position level (as matrix)
y = dataset.iloc[:, 2].values    # Salary (as array)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X, y)

# Creating predictions for all position levels
X_pred = np.arange(1, 11, 1).reshape(-1, 1)  # Position levels 1 through 10
y_pred = regressor.predict(X_pred)

# Creating comparison table
results = pd.DataFrame({
    'Position Level': X.flatten(),
    'Actual Salary': y,
    'Predicted Salary': y_pred,
    'Difference': y - y_pred,
    'Percentage Error': ((y - y_pred)/y)*100
})

print("\n=== Results Estimation ===")
print(results.round(2))
print("\nAverage Absolute Percentage Error:", 
      np.mean(np.abs(results['Percentage Error'])).round(2), "%")

# Predicting a specific new result
position_level = 6.5
predicted_salary = regressor.predict([[position_level]])
print(f"\nPredicted Salary for Position Level {position_level}: ${predicted_salary[0]:,.2f}")

# Visualizing the results
plt.figure(figsize=(12, 7))
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
plt.scatter(X, y, color='red', s=100, label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', 
         label='Random Forest Prediction')
plt.title('Salary vs Position Level\n(Random Forest Regression)', fontsize=16)
plt.xlabel('Position Level', fontsize=14)
plt.ylabel('Salary ($)', fontsize=14)

# Adding prediction points to the plot
for i, row in results.iterrows():
    plt.plot([row['Position Level'], row['Position Level']], 
             [row['Actual Salary'], row['Predicted Salary']], 
             'k--', alpha=0.3)
    plt.annotate(f"Δ${row['Difference']:,.0f}", 
                (row['Position Level'], (row['Actual Salary'] + row['Predicted Salary'])/2),
                textcoords="offset points", xytext=(0,5), ha='center')

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()