# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # Features: Age, Estimated Salary
y = dataset.iloc[:, 4].values       # Target: Purchased (0 or 1)

# Splitting the dataset (updated to model_selection)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling (crucial for SVM)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM with linear kernel
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicting and evaluating
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualization function (optimized)
def plot_decision_boundary(X_set, y_set, title):
    # Create mesh grid
    x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict and reshape
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    # Plot data points
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], 
                    X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), 
                    label=f'Class {j}')
    
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# Generate plots
plot_decision_boundary(X_train, y_train, 'SVM (Training set)')
plot_decision_boundary(X_test, y_test, 'SVM (Test set)')