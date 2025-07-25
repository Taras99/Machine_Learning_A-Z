# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Features matrix
y = dataset.iloc[:, -1].values   # Target vector

# Checking dataset
print("Dataset Head:")
print(dataset.head())
print("\nDataset Description:")
print(dataset.describe())
print("\nDataset Info:")
dataset.info()

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# One-hot encoding the first column (index 0)
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False), [0])  # drop='first' avoids dummy variable trap
    ],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Encoding the dependent variable (if categorical)
if isinstance(y[0], str):  # Only encode if target is string/categorical
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

# Validation of categorical encoding
print("\nEncoded Features:")
print(X[:5])  # Display first 5 rows of encoded features

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Scale y if it's continuous/numeric
if not isinstance(y[0], str):
    y_train = y_train.reshape(-1, 1)  # Reshape for scaler
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train).flatten()
    y_test = y_test.reshape(-1, 1)
    y_test = sc_y.transform(y_test).flatten()