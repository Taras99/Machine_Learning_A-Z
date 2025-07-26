# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('D:/Cources/Machine_Learning_A-Z/MultipleLinearRegression/50_Startups.csv') # Adjust full path as needed

# 1. Separate features and target (ensure y is numeric)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  # Make sure this is numeric

# Convert y to float if needed
y = y.astype(float)

# 2. Handle categorical data (safer implementation)
categorical_features = [3]  # Index of categorical column
numeric_features = [i for i in range(X.shape[1]) if i not in categorical_features]

ct = ColumnTransformer(
    [('one_hot', OneHotEncoder(drop='first'), categorical_features)],
    remainder='passthrough'
)

X_encoded = ct.fit_transform(X)

# 3. Convert to dense array if sparse (common after one-hot encoding)
if hasattr(X_encoded, "toarray"):
    X_encoded = X_encoded.toarray()

# 4. Ensure all values are finite and numeric
X_encoded = np.nan_to_num(X_encoded.astype(float))

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# 6. Statsmodels implementation (with proper constant)
X_with_const = sm.add_constant(X_train)  # Better than np.append

# 7. Backward elimination function
def backward_elimination(X, y, threshold=0.05):
    features = list(range(X.shape[1]))
    while True:
        X_with_const = sm.add_constant(X[:, features])
        model = sm.OLS(y, X_with_const).fit()
        p_values = model.pvalues[1:]  # Skip intercept
        max_p = max(p_values)
        
        if max_p > threshold:
            worst_feature = np.argmax(p_values)
            features.pop(worst_feature)
        else:
            break
    return features

# Get optimal features
optimal_features = backward_elimination(X_train, y_train)
print(f"Optimal features indices: {optimal_features}")

# Final model
final_X = sm.add_constant(X_train[:, optimal_features])
final_model = sm.OLS(y_train, final_X).fit()
print(final_model.summary())