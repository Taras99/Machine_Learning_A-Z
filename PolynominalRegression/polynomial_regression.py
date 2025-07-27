# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Note: We don't split into train/test because we have very few data points (10)
# and we want to understand the relationship across all levels

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Regression')
plt.title('Salary vs Position Level (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True)
plt.show()

# Visualising the Polynomial Regression results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue', 
         label=f'Polynomial Regression (degree={poly_reg.degree})')
plt.title('Salary vs Position Level (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True)
plt.show()

# Visualising with higher resolution and smoother curve
plt.figure(figsize=(10, 6))
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), 
         color='blue', label=f'Polynomial Regression (degree={poly_reg.degree})')
plt.title('Salary vs Position Level (High Resolution Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True)
plt.show()

# Predicting a new result with Linear Regression
level = 6.5
lin_pred = lin_reg.predict([[level]])
print(f"Linear Regression prediction for level {level}: ${lin_pred[0]:,.2f}")

# Predicting a new result with Polynomial Regression
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[level]]))
print(f"Polynomial Regression prediction for level {level}: ${poly_pred[0]:,.2f}")