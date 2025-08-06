# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Part 1 - Data Preprocessing

# Load the dataset
dataset = pd.read_csv('/Churn_Modelling.csv')

# Extract features and target
X = dataset.iloc[:, 3:13].values  # Columns from CreditScore to EstimatedSalary
y = dataset.iloc[:, 13].values    # Exited column

# Encoding categorical data
# 1. Label Encoding for Gender (column 2)
gender_encoder = LabelEncoder()
X[:, 2] = gender_encoder.fit_transform(X[:, 2])  # Male: 1, Female: 0

# 2. One-Hot Encoding for Geography (column 1)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [1])],  # drops first column to avoid dummy variable trap
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Feature Scaling (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Part 2 - Building the ANN

# Initialize the ANN
model = Sequential()

# Add input layer and first hidden layer with dropout
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))  # Helps prevent overfitting

# Add second hidden layer
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

# Add output layer (sigmoid for binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the ANN
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the ANN
history = model.fit(
    X_train, y_train,
    validation_split=0.1,  # Use 10% of training data for validation
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

# Part 3 - Evaluating the Model

# Predict test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate performance
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))